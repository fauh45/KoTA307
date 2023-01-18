import pandas as pd
import io

from dagster import InputContext, OutputContext, io_manager, IOManager

from kota_data_processing.resources.s3 import S3ObjectWrapper


class DataFrameS3CSVIOManage(IOManager):
    def __init__(self, s3obj: S3ObjectWrapper) -> None:
        super().__init__()

        self.s3obj = s3obj

    def _generate_name(self, name: str):
        return f"Output/{name}"

    def handle_output(
        self, context: "OutputContext", obj: pd.DataFrame
    ) -> None:
        stream = io.StringIO()
        obj.to_csv(stream)

        self.s3obj.upload(self._generate_name(context.name), stream.getvalue())

    def load_input(self, context: "InputContext") -> pd.DataFrame:
        return pd.read_csv(self.s3obj.download(self._generate_name(context.upstream_output.name)))  # type: ignore


@io_manager(required_resource_keys={"s3"})
def dataframe_s3_csv_io_manager(context):
    return DataFrameS3CSVIOManage(context.resources.s3)
