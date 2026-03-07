"""A collection of helper functions to infer an expanded schema from a datafile.

In addition to inferring a schema, a function is provided to validate an existing schema \
against allowed values and types. There is also a function to set monitoring parameters \
given a valid schema as input.
"""

import json
import logging
import math
import numbers
from typing import Dict, List, Union

# Third party packages
import pandas

# import avro.schema

logger = logging.getLogger(__name__)


def type_map(dtypes: List[str]) -> str:
    """
    Map Python data types to Avro types.

    Args:
        dtypes (List[str]): A list of Python data types.

    Returns:
        Corresponding Avro type, allowing for unions.
    """

    type_list = []

    if int in dtypes:
        type_list.append("int")
    if float in dtypes:
        type_list.append("float")
    if str in dtypes:
        type_list.append("string")
    if bool in dtypes:
        type_list.append("boolean")
    if type(None) in dtypes:
        type_list.append("null")

    # if dtypes is a list of one type only, return that type
    # (not in a list)
    if len(type_list) == 1:
        return type_list[0]
    return type_list


def data_class_map(avro_data_type: Union[str, List[str]]) -> str:
    """
    Map Avro types to dataClass ('numerical' or 'categorical').

    Args:
        avro_data_type (str or List[str]): Avro data type(s).

    Returns:
        'numerical' or 'categorical'.
    """

    # If input is a string, change to a list consisting of that string
    if not isinstance(avro_data_type, list):
        avro_data_type = [avro_data_type]

    # If a 'string' or 'boolean' is present, this takes precedence
    # (set as 'categorical')
    if ("string" in avro_data_type) or ("boolean" in avro_data_type):
        return "categorical"

    # A combination of floats, ints, and nulls returns "numerical"
    if (
        ("double" in avro_data_type)
        or ("float" in avro_data_type)
        or ("int" in avro_data_type)
        or ("long" in avro_data_type)
    ):
        return "numerical"

    # If the only data_type is "null", set as "categorical"
    if avro_data_type == ["null"]:
        return "categorical"

    logger.warning("No primitive avro type found! Setting to default: 'categorical'")
    return "categorical"


def role_map(field_name: str) -> str:
    """
    Map DataFrame fields to roles.

    Args:
        field_name (str): DataFrame column name.

    Returns:
        'label', 'score', 'predictor' or 'identifier'.
    """

    # Remove capitalization
    field_name = field_name.lower()

    # 'label', 'score', and 'identifier' roles are auto-detected
    # if using reserved keywords
    if field_name in ["label", "ground_truth"]:
        return "label"

    if field_name in ["score", "prediction"]:
        return "score"

    if field_name in ["id", "uuid"]:
        return "identifier"

    # Anything else is a predictor, including 'weight' columns, if any
    return "predictor"


def protected_class_map(field_name: str) -> bool:
    """
    Map DataFrame fields to protectedClass (boolean).

    Args:
        field_name (str): DataFrame column name.

    Returns:
        True if field_name is a protected class, False otherwise.
    """

    # Remove capitalization
    field_name = field_name.lower()

    return bool(
        field_name.lower()
        in [
            "race",
            "color",
            "religion",
            "sex",
            "gender",
            "pregnancy",
            "sexual_orientation",
            "gender_identity",
            "national_origin",
            "age",
            "disability",
        ]
    )


def drift_candidate_map(role: str) -> bool:
    """
    Map DataFrame field role to driftCandidate (boolean).

    Args:
        role (str): DataFrame field role.

    Returns:
        False if role is 'non_predictor', 'identifier', or 'weight'; True otherwise.
    """

    return bool(role not in ["non_predictor", "identifier", "weight"])


def scoring_optional_map(field_name: str) -> bool:
    """
    Map DataFrame fields to scoringOptional (Boolean)

    Args:
        field_name (str): DataFrame field.

    Returns:
        True if field role is 'label', 'score', or 'weight' or field is a protectedClass
        (not necessarily present at scoring time); False otherwise.
    """

    # Remove capitalization
    field_name = field_name.lower()

    return bool(
        role_map(field_name=field_name) in ["label", "score", "weight"]
        or protected_class_map(field_name=field_name)
    )


# Expanded Schema metadata
metadata_values = {
    "type": ["int", "float", "double", "long", "string", "boolean", "null"],
    "dataClass": ["numerical", "categorical"],
    "role": ["label", "score", "predictor", "non_predictor", "weight", "identifier", "prediction_date"],
    "protectedClass": [True, False],
    "driftCandidate": [True, False],
    "scoringOptional": [True, False],
}


def infer_schema(data: Union[str, pandas.DataFrame], schema_name: str) -> dict:
    """
    A function to infer an expanded schema definition from input DataFrame or file.

    Args:
        data (str or pandas.DataFrame): input filename or input DataFrame.

        schema_name (str): name to assign to resulting schema.

    Return:
        Schema definition (dict) containing metadata for all input data fields,
        including the following defaults:

            * 'specialValues': []
    """

    if isinstance(data, str):  # data is given as a filename
        dataframe = pandas.read_json(data, orient="records", lines=True)
        # Necessary step to keep pandas from casting Nones as np.nan (treated as float)
        dataframe = dataframe.where(pandas.notnull(dataframe), None)

    elif isinstance(data, pandas.DataFrame):  # data is passed as a DataFrame
        # Necessary step to keep pandas from casting Nones as np.nan (treated as float)
        dataframe = data.where(pandas.notnull(data), None)
    else:
        raise ValueError("data must be either a filename of a pandas.DataFrame.")

    # Schema fields
    fields = []

    # Python data types that will be mapped to Avro primitive types
    primitive_types = [str, int, float, bool, type(None)]

    for field in dataframe.columns.values:

        # Get all unique Python types in column
        unique_types = dataframe[field].map(type).unique()

        if set(unique_types) == set([float, type(None)]):
            not_null_values = dataframe[field][dataframe[field].notna()]
            integer_values = pandas.Series(
                [True if i.is_integer() else False for i in not_null_values]
            )
            # Check if all floats are in fact integers
            if integer_values.all():
                unique_types = [int, type(None)]

        # Map unique_types to primitive types (e.g. <class 'str'> -> str)
        dtypes = [i for i in primitive_types if i in unique_types]

        # Map Pandas dtypes to AVRO types
        field_avro_type = type_map(dtypes=dtypes)

        # Map field to a role
        field_role = role_map(field_name=field)

        # Map field_avro_type to a dataClass, with one exception (binary output)
        field_data_class = data_class_map(avro_data_type=field_avro_type)

        # 'score' and 'label' columns are considered categorical
        # if they take values in [0,1]
        if field_role in ["score", "label"]:
            if field_data_class == "numerical" and dataframe[field].isin([0, 1]).all():

                field_data_class = "categorical"

        fields.append(
            {
                "name": field,
                "type": field_avro_type,
                "dataClass": field_data_class,
                "role": field_role,
                "protectedClass": protected_class_map(field_name=field),
                "driftCandidate": drift_candidate_map(role=field_role),
                "specialValues": [],
                "scoringOptional": scoring_optional_map(field_name=field),
            }
        )

    schema_json = {
        "name": schema_name,
        "type": "record",
        "fields": fields,
    }

    return schema_json


def fail_on_invalid_schema(dataframe: pandas.DataFrame):
    """
    A function to validate a schema against allowed metadata values.

    Args:
        dataframe (pandas.DataFrame): Input dataframe corresponding to 'fields' value
            of schema definition dict.

    Raises:
        valueError: If schema definition is invalid.
    """

    for field in dataframe.index:
        for metadata in [
            "dataClass",
            "role",
            "protectedClass",
            "driftCandidate",
            "scoringOptional",
        ]:
            try:
                # Check if values are acceptable
                acceptable_values = (
                    dataframe.loc[field, metadata] not in metadata_values[metadata]
                )
            except Exception as ex:
                raise Exception("Missing property in extended schema: " + str(ex))
            if acceptable_values:
                raise ValueError(
                    "ValueError encountered while validating field {}: {} = {} not in {}".format(
                        field,
                        metadata,
                        dataframe.loc[field, metadata],
                        metadata_values[metadata],
                    )
                )
        field_types = dataframe.loc[field, "type"]

        # Check types are valid
        if not isinstance(field_types, list):
            field_types = [field_types]

        for primitive_type in field_types:
            if primitive_type not in metadata_values["type"]:
                raise ValueError(
                    "ValueError encountered while validating field {}: type = {} not in {}".format(
                        field,
                        primitive_type,
                        metadata_values["type"],
                    )
                )

        # Check specialValues are valid
        special_values = dataframe.loc[field, "specialValues"]

        # Check that specialValues are lists
        if not isinstance(special_values, List):
            raise TypeError(
                (
                    "TypeError encountered while validating field {}: "
                    + "Expected 'specialValues' to be of type <List>, got {}"
                ).format(field, type(special_values))
            )

        if special_values != []:  # Default specialValues
            # Check that specialValues is an array of dicts
            if not all(isinstance(element, Dict) for element in special_values):
                raise TypeError(
                    (
                        "TypeError encountered while validating field {}: "
                        + "All elements of 'specialValues' must be of type <dict>"
                    ).format(field)
                )

            # Check that the dicts have the same keys
            for special_values_dict in special_values:
                if set(special_values_dict.keys()) != set(["values", "purpose"]):
                    raise KeyError(
                        (
                            "KeyError encountered while validating field {}: "
                            + "Expected dictionaries in 'specialValues' array to "
                            + "have the keys ['values', 'purpose'], got {}"
                        ).format(field, list(special_values_dict.keys()))
                    )
                # check that the dicts have the correct value types
                if not isinstance(special_values_dict["values"], List):
                    raise TypeError(
                        (
                            "TypeError encountered while validating field {}: "
                            + "Expected specialValues['values'] to be of type <List>, got {}"
                        ).format(field, type(special_values_dict["values"]))
                    )
                if not isinstance(special_values_dict["purpose"], str):
                    raise TypeError(
                        (
                            "TypeError encountered while validating field {}: "
                            + "Expected specialValues['purpose'] to be of type <str>, got {}"
                        ).format(field, type(special_values_dict["purpose"]))
                    )


def deal_with_role_not_equal_one(columns_to_check: dict) -> dict:
    """A function to to check if a list from monitoring parameters,
    such as score_column, label_column, or weight_column, has more than 1 element
    (column name) or no elements at all.


    Args:
        columns_to_check (dict): A dictionary or list names and value to be checked.

    Raises:
        ValueError: If more than  element found in a list.

    Returns:
        dict: Processed version of input dict.

    Examples:
        Raising ValueError if a list of a particular role has more than 1 element:

        >>> deal_with_role_not_equal_one(
        ...     {"score_column": ["score", "pred"]}
        ... )
        ValueError: Schema has more than 1 score_column.

        Setting an empty role list to None, while keeping a one-element list unchanged:

        >>> deal_with_role_not_equal_one(
        ...     {
        ...         "score_column": ["score"],
        ...         "label_column": []
        ...     }
        ... )
        {
            "score_column": "score",
            "label_column": None
        }
    """

    for col_name, col_value in columns_to_check.items():
        # Check for multiple columns. If more than 1 each, raise error.
        if len(col_value) > 1:
            raise ValueError("Schema has more than 1 {}.".format(col_name))

        # Extract column names if only one of each exists; otherwise, set to None
        columns_to_check[col_name] = col_value[0] if col_value else None

    return columns_to_check


def set_monitoring_parameters(schema_json: dict, check_schema: bool = True) -> dict:
    """
    A function to set parameters for detectors/monitors.

    Args:
        schema_json (dict): Expanded schema definition of input data.

        check_schema (bool): Flag to validate input schema or not.

    Returns:
        Map of detector/monitoring parameters.
    """

    # Extract fields info from schema object and turn into DataFrame
    schema_df = pandas.DataFrame(schema_json["fields"]).set_index("name")

    # Initialize outputs
    categorical_columns = []
    numerical_columns = []

    identifier_columns = []
    weight_columns = []
    score_columns = []
    label_columns = []
    date_columns = []
    output_type = None

    feature_dataclass = {}
    protected_classes = []
    special_values = {}
    positive_label = []

    if check_schema:
        fail_on_invalid_schema(dataframe=schema_df)

    # Schema is valid - iterate over fields and extract relevant info
    for field in schema_df.index.values:

        if schema_df.loc[field, "driftCandidate"] and (
            schema_df.loc[field, "role"] in ["predictor", "non-predictor"]
        ):

            if schema_df.loc[field, "dataClass"] == "categorical":
                categorical_columns.append(field)
            elif schema_df.loc[field, "dataClass"] == "numerical":
                numerical_columns.append(field)

            # Add special values
            special_values[field] = [
                values_dict["values"]
                for values_dict in schema_df.loc[field, "specialValues"]
            ]

            # Add data classes
            feature_dataclass[field] = schema_df.loc[field, "dataClass"]

        # output_type is determined by dataClass of score column
        if schema_df.loc[field, "role"] == "score":
            score_columns.append(field)
            output_type = schema_df.loc[field, "dataClass"]

            # Add special values
            special_values[field] = [
                values_dict["values"]
                for values_dict in schema_df.loc[field, "specialValues"]
            ]

            feature_dataclass[field] = schema_df.loc[field, "dataClass"]

        elif schema_df.loc[field, "role"] == "label":
            label_columns.append(field)
            # may be overriding previously computed output_type
            output_type = schema_df.loc[field, "dataClass"]
            # Add special values
            special_values[field] = [
                values_dict["values"]
                for values_dict in schema_df.loc[field, "specialValues"]
            ]

            feature_dataclass[field] = schema_df.loc[field, "dataClass"]

            if 'positiveClassLabel' in schema_df.columns:
                value = schema_df.loc[field, 'positiveClassLabel']
                if value and (
                        not isinstance(value, numbers.Number) or
                        not math.isnan(value)):
                    positive_label.append(value)

        elif schema_df.loc[field, "role"] == "identifier":
            identifier_columns.append(field)

        elif schema_df.loc[field, "role"] == "weight":
            weight_columns.append(field)

        elif schema_df.loc[field, "role"] == "prediction_date":
            date_columns.append(field)

        if schema_df.loc[field, "protectedClass"]:
            protected_classes.append(field)

    # Check for multiple score, label, and weight columns. If more than 1 each, raise error.
    # Extract score and label column names if only one of each exists; otherwise, set to None

    cols_to_check = deal_with_role_not_equal_one(
        columns_to_check={
            "score_column": score_columns,
            "label_column": label_columns,
            "weight_column": weight_columns,
            "date_column": date_columns
        }
    )

    return {
        "predictors": categorical_columns + numerical_columns,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "identifier_columns": identifier_columns,
        "weight_column": cols_to_check["weight_column"],
        "score_column": cols_to_check["score_column"],
        "label_column": cols_to_check["label_column"],
        "date_column": cols_to_check["date_column"],
        "output_type": output_type,
        "feature_dataclass": feature_dataclass,
        "protected_classes": protected_classes,
        "special_values": special_values,
        "positive_label": positive_label
    }


def extract_input_schema(job_json: dict) -> dict:
    """
    A function to traverse a MOC job JSON, and extract the input schema definition.

    Args:
        job_json (dict): json of MOC job.

    Returns:
         Input Schema definition (dict).
    """

    # Extract rawJson from job_json
    try:
        model_object = json.loads(job_json["rawJson"])
    except:
        model_object = json.load(job_json["rawJson"])

    # Extract input schema
    try:
        input_schemas = model_object["referenceModel"]["storedModel"]["modelMetaData"][
            "inputSchema"
        ]
    except Exception:
        logger.warning(
            "No input schema found on a reference storedModel. \
            Using base storedModel for input schema"
        )
        input_schemas = model_object["model"]["storedModel"]["modelMetaData"][
            "inputSchema"
        ]

    # Checking to see if any input schemas exist
    if len(input_schemas) != 1:
        raise ValueError(
            "Expected exactly 1 input schema, found {}".format(len(input_schemas))
        )

    # At this point there is exactly one schema in input_schemas
    return input_schemas[0]["schemaDefinition"]


def validate_schema(job_json: dict):
    """
    A function to traverse a MOC job JSON, and validate the input schema definition.
    If there is an issue reading the schema it will raise an error

    Args:
        job_json (dict): json of MOC job.

    """

    # Extract rawJson from job_json
    model_object = json.loads(job_json["rawJson"])

    # Extract input schema
    try:
        input_schemas = model_object["referenceModel"]["storedModel"]["modelMetaData"][
            "inputSchema"
        ]
    except Exception:
        logger.warning(
            "No input schema found on a reference storedModel. \
            Using base storedModel for input schema"
        )
        input_schemas = model_object["model"]["storedModel"]["modelMetaData"][
            "inputSchema"
        ]

    # Checking to see if any input schemas exist
    if len(input_schemas) != 1:
        raise ValueError(
            "Expected exactly 1 input schema, found {}".format(len(input_schemas))
        )

    # At this point there is exactly one schema in input_schemas
    return
