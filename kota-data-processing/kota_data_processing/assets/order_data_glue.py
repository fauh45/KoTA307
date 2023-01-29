import pandas as pd
import hashlib

from dagster import Backoff, RetryPolicy, asset
from kota_data_processing.helper.glue_description import (
    glue_order_data_with_description,
)
from kota_data_processing.helper.query_product_description import (
    invalidate_cache,
)
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
        s3obj.download(file_name)
        for file_name in raw_exported_data_list
        if file_name is not None
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

    # Remove all orders that ordered "Extra Shipping Cost"
    df = df[~df["Lineitem name"].str.contains("Extra Shipping Cost")]

    invalidate_cache()

    return df[["Email", "Lineitem name", "Lineitem sku"]]


@asset(description="Raw cleaned main dataset", group_name="main_dataset")
def raw_cleaned_main_dataset(cleaned_raw_order_data: pd.DataFrame):
    df = cleaned_raw_order_data

    # Count how may orders an Email Have
    order_count = df["Email"].value_counts()

    # Remove all the user which have order less than 2
    df = df[~df["Email"].isin(order_count[order_count < 3].index)]

    return df


@asset(
    required_resource_keys={"shopify"},
    description="Final main dataset",
    group_name="main_dataset",
    retry_policy=RetryPolicy(
        max_retries=5, delay=5, backoff=Backoff.EXPONENTIAL
    ),
    io_manager_def=dataframe_s3_csv_io_manager,
)
def cleaned_main_dataset(raw_cleaned_main_dataset: pd.DataFrame):
    return glue_order_data_with_description(raw_cleaned_main_dataset)


@asset(
    description="Raw cleaned transfer dataset", group_name="transfer_dataset"
)
def raw_cleaned_transfer_dataset(cleaned_raw_order_data: pd.DataFrame):
    df = cleaned_raw_order_data

    # Count how may orders an Email Have
    order_count = df["Email"].value_counts()

    # Remove all the user which have order less than 2
    df = df[df["Email"].isin(order_count[order_count == 2].index)]

    return df


@asset(
    required_resource_keys={"shopify"},
    description="Final transfer dataset",
    group_name="transfer_dataset",
    retry_policy=RetryPolicy(
        max_retries=5, delay=5, backoff=Backoff.EXPONENTIAL
    ),
    io_manager_def=dataframe_s3_csv_io_manager,
)
def cleaned_transfer_dataset(raw_cleaned_transfer_dataset: pd.DataFrame):
    return glue_order_data_with_description(raw_cleaned_transfer_dataset)
