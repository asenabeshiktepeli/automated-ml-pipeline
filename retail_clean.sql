@'
{{ config(materialized='table') }}

with base as (
    select
        invoice_no,
        stock_code,
        product,
        quantity,
        abs(quantity)                                          as quantity_abs,
        invoice_date,
        price,
        customer_id,
        region,
        quantity * price                                       as revenue,
        case when invoice_no like 'C%' then 1 else 0 end        as returned,
        extract(month from invoice_date)                        as month,
        extract(dow from invoice_date)                          as day_of_week,
        upper(split_part(trim(product), ' ', 1))                as category
    from {{ ref('stg_retail') }}
),

customer_features as (
    select
        *,
        count(*) over (
            partition by customer_id
            order by invoice_date, invoice_no, stock_code
            rows between unbounded preceding and 1 preceding
        ) as customer_prior_orders,

        avg(revenue) over (
            partition by customer_id
            order by invoice_date, invoice_no, stock_code
            rows between unbounded preceding and 1 preceding
        ) as customer_avg_order_value_raw,

        avg(returned) over (
            partition by customer_id
            order by invoice_date, invoice_no, stock_code
            rows between unbounded preceding and 1 preceding
        ) as customer_return_rate_raw,

        lag(invoice_date) over (
            partition by customer_id
            order by invoice_date, invoice_no, stock_code
        ) as previous_invoice_date
    from base
)

select
    invoice_no,
    stock_code,
    product,
    quantity,
    quantity_abs,
    invoice_date,
    price,
    customer_id,
    region,
    revenue,
    returned,
    month,
    day_of_week,
    category,
    coalesce(customer_prior_orders, 0)                                        as customer_prior_orders,
    coalesce(customer_avg_order_value_raw, 0.0)                               as customer_avg_order_value,
    coalesce(customer_return_rate_raw, 0.0)                                   as customer_return_rate,
    coalesce(date_diff('day', previous_invoice_date, invoice_date), 9999)     as customer_recency_days
from customer_features
'@ | Set-Content -Path "dbt_project\models\marts\retail_clean.sql" -Encoding utf8