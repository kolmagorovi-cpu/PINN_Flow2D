#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автоматический запуск и управление TensorBoard для проекта PINN Learning.
Содержит класс TensorBoardLauncher для запуска TensorBoard сервера в фоновом режиме.

Автор: Колмагоров И.Д.
Дата: 2026
"""

import os
import sys
import logging
import subprocess
import time
import socket
from typing import Optional
import threading

logger = logging.getLogger(__name__)


class TensorBoardLauncher:
    """
    Автоматический запуск и управление сервером TensorBoard.
    
    Запускает TensorBoard в фоновом потоке, проверяет доступность порта,
    предоставляет URL для доступа к веб-интерфейсу.
    """
    
    def __init__(self, log_dir: str = "logs", port: int = 6006, auto_launch: bool = True):
        """
        Args:
            log_dir: директория с логами TensorBoard
            port: порт для веб-интерфейса TensorBoard
            auto_launch: автоматически запускать TensorBoard
        """
        self.log_dir = os.path.abspath(log_dir)
        self.port = port
        self.auto_launch = auto_launch
        self.process: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self._is_running = False
        
    def is_port_available(self, port: Optional[int] = None) -> bool:
        """
        Проверяет, доступен ли указанный порт.
        
        Args:
            port: порт для проверки (по умолчанию self.port)
            
        Returns:
            True если порт доступен, False если уже занят
        """
        check_port = port if port is not None else self.port
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', check_port))
                return True
        except OSError:
            return False
    
    def find_free_port(self, start_port: int = 6006, max_attempts: int = 10) -> int:
        """
        Находит первый свободный порт начиная с start_port.
        
        Args:
            start_port: начальный порт для поиска
            max_attempts: максимальное количество попыток
            
        Returns:
            Номер свободного порта
            
        Raises:
            RuntimeError: если не удалось найти свободный порт
        """
        for attempt in range(max_attempts):
            port = start_port + attempt
            if self.is_port_available(port):
                logger.info(f"Найден свободный порт {port} для TensorBoard")
                return port
        
        raise RuntimeError(f"Не удалось найти свободный порт в диапазоне {start_port}-{start_port + max_attempts - 1}")
    
    def launch(self, wait_for_ready: bool = True, timeout: int = 10) -> bool:
        """
        Запускает TensorBoard в фоновом режиме.
        
        Args:
            wait_for_ready: ждать ли готовности сервера
            timeout: максимальное время ожидания готовности (секунды)
            
        Returns:
            True если TensorBoard успешно запущен, False в случае ошибки
        """
        if not self.auto_launch:
            logger.info("Автоматический запуск TensorBoard отключен в конфигурации")
            return False
        
        # Проверяем, существует ли директория логов
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
            logger.info(f"Создана директория для логов: {self.log_dir}")
        
        # Проверяем, есть ли логи
        if not os.listdir(self.log_dir):
            logger.info("Директория логов пуста. TensorBoard будет запущен, но данные появятся позже.")
        
        # Находим свободный порт
        if not self.is_port_available():
            logger.warning(f"Порт {self.port} занят, ищу свободный...")
            try:
                self.port = self.find_free_port(self.port)
            except RuntimeError as e:
                logger.error(f"Не удалось запустить TensorBoard: {e}")
                return False
        
        # Проверяем, установлен ли tensorboard
        try:
            import tensorboard
            logger.info(f"TensorBoard версии {tensorboard.__version__} найден")
        except ImportError:
            logger.error("TensorBoard не установлен. Выполните: pip install tensorboard")
            return False
        
        # Запускаем TensorBoard в фоновом потоке
        logger.info(f"Запуск TensorBoard на порту {self.port} с логами из {self.log_dir}")
        
        try:
            self.thread = threading.Thread(
                target=self._run_tensorboard,
                daemon=True,
                name="TensorBoard-Launcher"
            )
            self.thread.start()
            
            if wait_for_ready:
                if self._wait_for_ready(timeout):
                    logger.info(f"✅ TensorBoard успешно запущен")
                    logger.info(f"📊 Откройте браузер: http://localhost:{self.port}")
                    self._is_running = True
                    return True
                else:
                    logger.error(f"❌ TensorBoard не смог запуститься за {timeout} секунд")
                    return False
            else:
                self._is_running = True
                logger.info(f"TensorBoard запущен в фоновом режиме. Порт: {self.port}")
                return True
                
        except Exception as e:
            logger.error(f"Ошибка запуска TensorBoard: {e}")
            return False
    
    def _run_tensorboard(self):
        """
        Внутренний метод для запуска TensorBoard в отдельном потоке.
        """
        try:
            # Формируем команду
            cmd = [
                sys.executable, "-m", "tensorboard.main",
                "--logdir", self.log_dir,
                "--port", str(self.port),
                "--host", "localhost",
                "--reload_interval", "5",  # Проверка новых логов каждые 5 секунд
                "--purge_orphaned_data", "true",
                "--samples_per_plugin", "scalars=100000",  # Сохранять больше точек
            ]
            
            logger.debug(f"Команда TensorBoard: {' '.join(cmd)}")
            
            # Запускаем процесс
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            
            # Ждем завершения процесса (блокирует поток)
            self.process.wait()
            
            # Проверяем код возврата
            if self.process.returncode != 0:
                stderr = self.process.stderr.read().decode('utf-8', errors='replace')
                logger.error(f"TensorBoard завершился с кодом {self.process.returncode}: {stderr}")
            
            self._is_running = False
            logger.warning("Процесс TensorBoard завершен")
            
        except Exception as e:
            logger.error(f"Критическая ошибка в потоке TensorBoard: {e}")
            self._is_running = False
    
    def _wait_for_ready(self, timeout: int = 10) -> bool:
        """
        Ждет, пока TensorBoard станет доступен.
        
        Args:
            timeout: максимальное время ожидания (секунды)
            
        Returns:
            True если сервер готов, False если таймаут
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_port_available(self.port):
                # Порт еще занят, ждем
                time.sleep(0.5)
            else:
                # Порт освободился - сервер запущен
                return True
        
        return False
    
    def stop(self):
        """
        Останавливает процесс TensorBoard.
        """
        if self.process and self.process.poll() is None:
            logger.info("Остановка TensorBoard...")
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                logger.info("TensorBoard остановлен")
            except Exception as e:
                logger.error(f"Ошибка остановки TensorBoard: {e}")
        
        self._is_running = False
    
    @property
    def is_running(self) -> bool:
        """Проверяет, запущен ли TensorBoard."""
        if self.process:
            return self.process.poll() is None
        return False
    
    def get_url(self) -> str:
        """
        Возвращает URL для доступа к TensorBoard.
        
        Returns:
            Строка с URL
        """
        return f"http://localhost:{self.port}"
    
    def __del__(self):
        """Деструктор - останавливает TensorBoard при удалении объекта."""
        if self.is_running:
            self.stop()
