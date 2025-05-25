def total_revenue(df):
    return df['TotalAmount'].sum()

def average_total_amount(df):
    return df['TotalAmount'].mean()

def order_count_by_customer(df):
    return df.groupby('Customer').size()

def high_value_orders(df, threshold=500):
    return df[df['TotalAmount'] > threshold]

def orders_by_date_range(df, start, end):
    mask = (df['OrderDate'] >= start) & (df['OrderDate'] <= end)
    return df[mask]

def sorted_by_order_date(df, reverse=True):
    return df.sort_values(by='OrderDate', ascending=not reverse)

def group_by_category(df):
    return df.groupby('Category').agg({'Quantity': 'sum', 'TotalAmount': 'sum'})

def top_customers(df, top_n=3):
    return df.groupby('Customer')['TotalAmount'].sum().sort_values(ascending=False).head(top_n)
