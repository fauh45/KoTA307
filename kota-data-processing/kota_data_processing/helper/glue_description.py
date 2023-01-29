import pandas as pd
from kota_data_processing.helper.query_product_description import query_product_description


def glue_order_data_with_description(cleaned_raw_order_data: pd.DataFrame):
    df = cleaned_raw_order_data

    df["Product description"] = df.apply(
        lambda x: query_product_description(x["Lineitem sku"]), axis=1
    )
    df["Product description"] = df["Product description"].apply(
        lambda x: str(x).strip()
    )
    df = df[~(df["Product description"] == "")]

    return df