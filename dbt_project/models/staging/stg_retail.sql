{{ config(materialized='view') }}

select
    "InvoiceNo"    as invoice_no,
    "StockCode"    as stock_code,
    "Description"  as product,
    "Quantity"     as quantity,
    strptime("InvoiceDate", '%m/%d/%Y %H:%M') as invoice_date,
    "UnitPrice"    as price,
    "CustomerID"   as customer_id,
    "Country"      as region
from read_parquet('data/retail_data.parquet')
where "CustomerID" is not null
  and "Description" is not null
  and "UnitPrice" > 0
