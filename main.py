from src.data_processing import load_data
from src.analysis import *
from src.model import train_model, predict_total_amount

# Завантаження та обробка даних
df = load_data("data/orders.csv")

print("=== АНАЛІЗ ===")
print("Сумарний дохід:", total_revenue(df))
print("Середній TotalAmount:", average_total_amount(df))
print("Кількість замовлень по клієнтах:\n", order_count_by_customer(df))
print("Замовлення > 500:\n", high_value_orders(df))
print("Сортування за датою:\n", sorted_by_order_date(df))
print("Замовлення з 5 по 10 червня:\n", orders_by_date_range(df, '2023-06-05', '2023-06-10'))
print("Групування по категорії:\n", group_by_category(df))
print("ТОП-3 клієнтів:\n", top_customers(df))

print("=== ТРЕНУВАННЯ AI ===")
model = train_model(df)
print("Модель збережена.")

print("=== ПРОГНОЗ ===")
pred = predict_total_amount(2, 200)
print("Прогнозована сума покупки (2 штуки по 200):", pred)

from src.plot import (
    plot_order_count_by_date,
    plot_income_distribution_by_category
)

print("=== ПОБУДОВА ГРАФІКІВ ===")
plot_order_count_by_date(df)
plot_income_distribution_by_category(df)
