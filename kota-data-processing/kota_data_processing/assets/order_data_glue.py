import pandas as pd
import hashlib
import shopify
import json

from dagster import Backoff, RetryPolicy, asset, fs_io_manager
from kota_data_processing.resources.df_s3_csv_io_manager import (
    dataframe_s3_csv_io_manager,
)

from kota_data_processing.resources.s3 import S3ObjectWrapper


@asset(
    required_resource_keys={"s3"},
    description="List of raw files exported from Shopify Shop orders",
    group_name="raw_order_data",
)
def raw_exported_data_list(context):
    s3obj: S3ObjectWrapper = context.resources.s3
    exported_obj_list = s3obj.list(prefix="ExportedOrder/")

    return [files for files in exported_obj_list if files.endswith(".csv")]


@asset(
    required_resource_keys={"s3"},
    description="Merge together exported order data",
    group_name="raw_order_data",
)
def raw_merged_exported_order_data(context, raw_exported_data_list: list[str]):
    s3obj: S3ObjectWrapper = context.resources.s3

    downloaded_obj = [
        s3obj.download(file_name) for file_name in raw_exported_data_list
    ]
    read_csv_obj = [
        pd.read_csv(file.get("Body"))  # type: ignore
        for file in downloaded_obj
    ]

    return pd.concat(read_csv_obj)


@asset(
    description="Obfuscating, and cleaning up raw data of <1 order, and removing extra shipping cost orders",
    group_name="cleaned_order_data",
)
def cleaned_raw_order_data(raw_merged_exported_order_data: pd.DataFrame):
    df = raw_merged_exported_order_data

    # Obfuscating personal data
    df["Email"] = df["Email"].apply(
        lambda x: hashlib.sha1(str(x).encode()).hexdigest()
    )

    # Count how may orders an Email Have
    order_count = df["Email"].value_counts()

    # Remove all the user which have order less than 2
    df = df[~df["Email"].isin(order_count[order_count < 2].index)]

    # Remove all orders that ordered "Extra Shipping Cost"
    df = df[~df["Lineitem name"].str.contains("Extra Shipping Cost")]

    return df[["Email", "Lineitem name", "Lineitem sku"]]


@asset(
    required_resource_keys={"shopify"},
    description="List of all cleaned orders with products details on it",
    group_name="cleaned_order_data",
    retry_policy=RetryPolicy(
        max_retries=5, delay=5, backoff=Backoff.EXPONENTIAL
    ),
    io_manager_def=dataframe_s3_csv_io_manager,
)
def cleaned_order_data_products(cleaned_raw_order_data: pd.DataFrame):
    cache = dict()

    def _query_product_description(sku: str):
        if sku in cache:
            return cache[sku]

        graphql_query = (
            'query { products(first:1, query:"sku:%s") { edges { node { description } } } }'
            % sku
        )
        print(graphql_query)

        query_result = json.loads(shopify.GraphQL().execute(graphql_query))

        print(query_result)

        edges = query_result["data"]["products"]["edges"]
        description = ""
        if len(edges) < 1:
            return description

        description = edges[0]["node"]["description"]

        cache[sku] = description
        return description

    df = cleaned_raw_order_data

    df["Product description"] = df.apply(
        lambda x: _query_product_description(x["Lineitem sku"]), axis=1
    )
    df = df[~(df["Product description"] == "")]

    return df
