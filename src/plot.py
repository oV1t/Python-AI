import matplotlib.pyplot as plt

# 1. Графік кількості замовлень по датах
def plot_order_count_by_date(df):
    order_counts = df.groupby('OrderDate').size()

    plt.figure(figsize=(8, 5))
    order_counts.plot(kind='line', marker='o', color='teal')
    plt.title('Кількість замовлень по датах')
    plt.xlabel('Дата')
    plt.ylabel('Кількість замовлень')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

# 2. Діаграма розподілу доходів по категоріях
def plot_income_distribution_by_category(df):
    income_by_category = df.groupby('Category')['TotalAmount'].sum()

    plt.figure(figsize=(6, 6))
    income_by_category.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title('Розподіл доходів по категоріях')
    plt.ylabel('')  # Прибираємо підпис осі Y
    plt.tight_layout()
    plt.show()
