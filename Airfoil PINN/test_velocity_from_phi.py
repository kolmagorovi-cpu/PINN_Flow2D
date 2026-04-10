#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для проверки корректности вычисления скоростей из потенциала.
"""

import torch
import sys
import os

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import PotentialPINN

def test_velocity_computation():
    """Проверяем, что u = ∂φ/∂x вычисляется правильно"""
    
    print("=" * 60)
    print("ТЕСТ: Вычисление скоростей из потенциала")
    print("=" * 60)
    
    # Создаём простую модель
    input_dim = 4  # x, y, sdf, boundary_one_hot
    model = PotentialPINN(
        input_dim=input_dim,
        layers=[32, 32, 32],
        activation='tanh',
        output_bias=0.0
    )
    model.eval()
    
    # Создаём тестовые данные
    batch_size = 10
    x = torch.randn(batch_size, 1, requires_grad=True)
    y = torch.randn(batch_size, 1, requires_grad=True)
    sdf = torch.rand(batch_size, 1) * 10
    bt = torch.zeros(batch_size, 1)
    
    # Формируем входной тензор
    input_tensor = torch.cat([x, y, sdf, bt], dim=1)
    
    # Прямой проход
    phi = model(input_tensor)
    
    print(f"\nВходные данные:")
    print(f"  x: {x[0, 0].item():.4f}")
    print(f"  y: {y[0, 0].item():.4f}")
    print(f"  φ: {phi[0, 0].item():.6f}")
    
    # Вычисляем градиенты
    phi_x = torch.autograd.grad(
        phi, x, grad_outputs=torch.ones_like(phi),
        create_graph=True, retain_graph=True
    )[0]
    
    phi_y = torch.autograd.grad(
        phi, y, grad_outputs=torch.ones_like(phi),
        create_graph=True, retain_graph=True
    )[0]
    
    print(f"\nСкорости (через градиенты):")
    print(f"  u = ∂φ/∂x: {phi_x[0, 0].item():.6f}")
    print(f"  v = ∂φ/∂y: {phi_y[0, 0].item():.6f}")
    
    # Тест с известным потенциалом: φ = U_inf * x
    # Тогда u = U_inf, v = 0
    print("\n" + "=" * 60)
    print("ТЕСТ: Проверка на линейном потенциале φ = 2.5 * x")
    print("=" * 60)
    
    # Создаём "фейковую" модель, которая возвращает φ = 2.5 * x
    class FakeModel(torch.nn.Module):
        def forward(self, input_tensor):
            return 2.5 * input_tensor[:, 0:1]
    
    fake_model = FakeModel()
    test_input = torch.tensor([[1.0, 0.5, 5.0, 0.0], [2.0, 1.0, 3.0, 0.0]], requires_grad=True)
    x_test = test_input[:, 0:1]
    
    phi_test = fake_model(test_input)
    u_test = torch.autograd.grad(phi_test, x_test, grad_outputs=torch.ones_like(phi_test),
                                  create_graph=False)[0]
    
    print(f"\nОжидаемо: u = 2.5, v = 0")
    print(f"Получено: u = {u_test[0, 0].item():.6f}")
    print(f"Получено: u = {u_test[1, 0].item():.6f}")
    
    if abs(u_test[0, 0].item() - 2.5) < 1e-6:
        print("\n✅ ТЕСТ ПРОЙДЕН: Градиенты вычисляются правильно!")
    else:
        print("\n❌ ТЕСТ ПРОВАЛЕН: Ошибка в градиентах!")
    
    print("\n" + "=" * 60)
    print("ТЕСТ: Проверка в режиме no_grad (для inference)")
    print("=" * 60)
    
    with torch.no_grad():
        # В no_grad режиме нужно явно включить grad для x, y
        x_ng = torch.randn(5, 1, requires_grad=True)
        y_ng = torch.randn(5, 1, requires_grad=True)
        sdf_ng = torch.rand(5, 1) * 10
        bt_ng = torch.zeros(5, 1)
        input_ng = torch.cat([x_ng, y_ng, sdf_ng, bt_ng], dim=1)
        
        phi_ng = model(input_ng)
        
        # Градиенты должны работать, т.к. x_ng, y_ng имеют requires_grad=True
        u_ng = torch.autograd.grad(
            phi_ng, x_ng, grad_outputs=torch.ones_like(phi_ng),
            create_graph=False, retain_graph=False, allow_unused=True
        )[0]
        
        if u_ng is not None:
            print(f"✅ В no_grad режиме градиенты работают")
            print(f"   u[0] = {u_ng[0, 0].item():.6f}")
        else:
            print(f"❌ В no_grad режиме градиенты = None!")

if __name__ == '__main__':
    test_velocity_computation()
