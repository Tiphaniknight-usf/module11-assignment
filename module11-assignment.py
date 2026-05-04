# Module 11 Assignment: Data Visualization with Matplotlib
# SunCoast Retail Visual Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("SUNCOAST RETAIL VISUAL ANALYSIS")
print("=" * 60)

# ----- DATA CREATION (DO NOT MODIFY) -----
np.random.seed(42)

quarters = pd.date_range(start='2022-01-01', periods=8, freq='QE')
quarter_labels = ['Q1 2022', 'Q2 2022', 'Q3 2022', 'Q4 2022',
                 'Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023']

locations = ['Tampa', 'Miami', 'Orlando', 'Jacksonville']
categories = ['Electronics', 'Clothing', 'Home Goods', 'Sporting Goods', 'Beauty']

quarterly_data = []

for quarter_idx, quarter in enumerate(quarters):
    for location in locations:
        for category in categories:
            base_sales = np.random.normal(loc=100000, scale=20000)

            seasonal_factor = 1.3 if quarter.quarter == 4 else (0.8 if quarter.quarter == 1 else 1.0)

            location_factor = {
                'Tampa': 1.0, 'Miami': 1.2, 'Orlando': 0.9, 'Jacksonville': 0.8
            }[location]

            category_factor = {
                'Electronics': 1.5, 'Clothing': 1.0, 'Home Goods': 0.8,
                'Sporting Goods': 0.7, 'Beauty': 0.9
            }[category]

            growth_factor = (1 + 0.05/4) ** quarter_idx
            sales = base_sales * seasonal_factor * location_factor * category_factor * growth_factor
            sales *= np.random.normal(loc=1.0, scale=0.1)

            ad_spend = (sales ** 0.7) * 0.05 * np.random.normal(loc=1.0, scale=0.2)

            quarterly_data.append({
                'Quarter': quarter,
                'QuarterLabel': quarter_labels[quarter_idx],
                'Location': location,
                'Category': category,
                'Sales': round(sales, 2),
                'AdSpend': round(ad_spend, 2),
                'Year': quarter.year
            })

customer_data = []
total_customers = 2000

age_params = {
    'Tampa': (45, 15),
    'Miami': (35, 12),
    'Orlando': (38, 14),
    'Jacksonville': (42, 13)
}

for location in locations:
    mean_age, std_age = age_params[location]
    customer_count = int(total_customers * {
        'Tampa': 0.3, 'Miami': 0.35, 'Orlando': 0.2, 'Jacksonville': 0.15
    }[location])

    ages = np.clip(np.random.normal(mean_age, std_age, customer_count), 18, 80).astype(int)

    for age in ages:
        if age < 30:
            category_preference = np.random.choice(categories, p=[0.3,0.3,0.1,0.2,0.1])
        elif age < 50:
            category_preference = np.random.choice(categories, p=[0.25,0.2,0.25,0.15,0.15])
        else:
            category_preference = np.random.choice(categories, p=[0.15,0.1,0.35,0.1,0.3])

        base_amount = np.random.gamma(5,20)
        price_tier = np.random.choice(['Budget','Mid-range','Premium'], p=[0.3,0.5,0.2])
        tier_factor = {'Budget':0.7,'Mid-range':1.0,'Premium':1.8}[price_tier]

        customer_data.append({
            'Location': location,
            'Age': age,
            'Category': category_preference,
            'PurchaseAmount': round(base_amount * tier_factor,2),
            'PriceTier': price_tier
        })

sales_df = pd.DataFrame(quarterly_data)
customer_df = pd.DataFrame(customer_data)

sales_df['Quarter_Num'] = sales_df['Quarter'].dt.quarter
sales_df['SalesPerDollarSpent'] = sales_df['Sales'] / sales_df['AdSpend']

# ----- VISUALIZATION FUNCTIONS -----

def plot_quarterly_sales_trend():
    data = sales_df.groupby('QuarterLabel')['Sales'].sum()
    fig, ax = plt.subplots(figsize=(10,6))
    data.plot(marker='o', ax=ax)
    ax.set_title('Overall Quarterly Sales Trend')
    ax.grid(True)
    return fig

def plot_location_sales_comparison():
    data = sales_df.groupby(['QuarterLabel','Location'])['Sales'].sum().unstack()
    fig, ax = plt.subplots(figsize=(10,6))
    data.plot(marker='o', ax=ax)
    ax.set_title('Quarterly Sales by Location')
    ax.grid(True)
    return fig

def plot_category_performance_by_location():
    latest = sales_df['QuarterLabel'].iloc[-1]
    data = sales_df[sales_df['QuarterLabel']==latest]
    grouped = data.groupby(['Location','Category'])['Sales'].sum().unstack()
    fig, ax = plt.subplots(figsize=(10,6))
    grouped.plot(kind='bar', ax=ax)
    ax.set_title(f'Category Performance - {latest}')
    return fig

def plot_sales_composition_by_location():
    grouped = sales_df.groupby(['Location','Category'])['Sales'].sum().unstack()
    percent = grouped.div(grouped.sum(axis=1), axis=0)*100
    fig, ax = plt.subplots(figsize=(10,6))
    percent.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Sales Composition by Location (%)')
    return fig

def plot_ad_spend_vs_sales():
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(sales_df['AdSpend'], sales_df['Sales'])
    m,b = np.polyfit(sales_df['AdSpend'], sales_df['Sales'],1)
    ax.plot(sales_df['AdSpend'], m*sales_df['AdSpend']+b)
    ax.set_title('Ad Spend vs Sales')
    return fig

def plot_ad_efficiency_over_time():
    data = sales_df.groupby('QuarterLabel')['SalesPerDollarSpent'].mean()
    fig, ax = plt.subplots(figsize=(10,6))
    data.plot(marker='o', ax=ax)
    ax.set_title('Advertising Efficiency Over Time')
    ax.grid(True)
    return fig

def plot_customer_age_distribution():
    fig, axes = plt.subplots(2,3, figsize=(15,8))
    axes = axes.flatten()
    axes[0].hist(customer_df['Age'], bins=15)
    axes[0].set_title('Overall Age')
    for i, loc in enumerate(customer_df['Location'].unique(),1):
        subset = customer_df[customer_df['Location']==loc]
        axes[i].hist(subset['Age'], bins=15)
        axes[i].set_title(loc)
    return fig

def plot_purchase_by_age_group():
    bins=[18,30,45,60,80]
    labels=['18-30','31-45','46-60','61+']
    customer_df['AgeGroup']=pd.cut(customer_df['Age'],bins=bins,labels=labels)
    fig, ax = plt.subplots(figsize=(10,6))
    customer_df.boxplot(column='PurchaseAmount', by='AgeGroup', ax=ax)
    plt.suptitle('')
    return fig

def plot_purchase_amount_distribution():
    fig, ax = plt.subplots(figsize=(10,6))
    ax.hist(customer_df['PurchaseAmount'], bins=20)
    ax.set_title('Purchase Amount Distribution')
    return fig

def plot_sales_by_price_tier():
    totals = customer_df.groupby('PriceTier')['PurchaseAmount'].sum()
    explode=[0.1 if x==totals.max() else 0 for x in totals]
    fig, ax = plt.subplots(figsize=(8,8))
    ax.pie(totals, labels=totals.index, autopct='%1.1f%%', explode=explode)
    return fig

def plot_category_market_share():
    totals = sales_df.groupby('Category')['Sales'].sum()
    explode=[0.1 if x==totals.max() else 0 for x in totals]
    fig, ax = plt.subplots(figsize=(8,8))
    ax.pie(totals, labels=totals.index, autopct='%1.1f%%', explode=explode)
    return fig

def plot_location_sales_distribution():
    totals = sales_df.groupby('Location')['Sales'].sum()
    fig, ax = plt.subplots(figsize=(8,8))
    ax.pie(totals, labels=totals.index, autopct='%1.1f%%')
    return fig

def create_business_dashboard():
    fig, axes = plt.subplots(2,2, figsize=(15,10))
    sales_df.groupby('QuarterLabel')['Sales'].sum().plot(ax=axes[0,0], marker='o')
    axes[0,0].set_title('Sales Trend')
    sales_df.groupby('Category')['Sales'].sum().plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Category Sales')
    sales_df.groupby('Location')['Sales'].sum().plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Location Sales')
    sales_df.groupby('QuarterLabel')['SalesPerDollarSpent'].mean().plot(ax=axes[1,1], marker='o')
    axes[1,1].set_title('Ad Efficiency')
    fig.suptitle('SunCoast Retail Dashboard')
    return fig

def main():
    fig1 = plot_quarterly_sales_trend()
    fig2 = plot_location_sales_comparison()
    fig3 = plot_category_performance_by_location()
    fig4 = plot_sales_composition_by_location()
    fig5 = plot_ad_spend_vs_sales()
    fig6 = plot_ad_efficiency_over_time()
    fig7 = plot_customer_age_distribution()
    fig8 = plot_purchase_by_age_group()
    fig9 = plot_purchase_amount_distribution()
    fig10 = plot_sales_by_price_tier()
    fig11 = plot_category_market_share()
    fig12 = plot_location_sales_distribution()
    fig13 = create_business_dashboard()

    print("""
KEY BUSINESS INSIGHTS:
1. Sales peak every Q4 and grow steadily year over year.
2. Miami consistently generates the highest revenue.
3. Electronics dominate overall sales.
4. Advertising strongly correlates with higher sales.
5. Ages 31–60 spend the most.
6. Mid-range pricing generates the most revenue.
""")

    plt.show()

if __name__ == "__main__":
    main()