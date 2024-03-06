import collections
from deepdiff import DeepDiff
from collections import Counter, defaultdict
import re
import operator
import pandas as pd
import numpy as np


def process_step_per_order(df, case_id, step_name):
    """For each case id, provides a list of steps completed in the right order
    Returns:
        pd.DataFrame
    """
    process_per_order = (
        df[[case_id, step_name]]
        .set_index(case_id)
        .groupby(case_id)
        .apply(lambda x: x.to_numpy().tolist())
        .to_dict()
    )
    df_process = pd.DataFrame(
        list(process_per_order.items()), columns=[case_id, "Nested_steps"]
    )
    df_process["Steps"] = df_process["Nested_steps"].apply(
        lambda x: [item for items in x for item in items]
    )
    df_process = df_process.drop("Nested_steps", axis=1)
    return df_process


def remove_item_dico(df):
    try:
        del df["dictionary_item_removed"]
        return df
    except:
        return df


def compliance_check(df, process_flow):
    """Returns a dataframe with analysis of the different variants of the process flow existing
    Returns:
        pd.DataFrame
    """
    df_flow = pd.DataFrame(df["Steps"].value_counts())
    df_flow.reset_index(inplace=True)
    print(f"There are {len(df_flow)} variations of the process.")

    df_flow["Duplicated_steps"] = df_flow["Steps"].apply(
        lambda x: [item for item, count in collections.Counter(x).items() if count > 1]
    )
    df_flow["Missing_steps"] = df_flow["Steps"].apply(
        lambda x: list(set(process_flow).difference(x))
    )
    df_flow["Reversed_steps"] = df_flow["Steps"].apply(
        lambda x: DeepDiff(
            dict(enumerate(process_flow)), dict(enumerate(list(dict.fromkeys(x))))
        )
    )
    # df_flow['step_reverse'] = df_flow['Reversed_steps'].apply(lambda x : x['values_changed'])
    df_flow["Rejected"] = df_flow["Steps"].apply(
        lambda x: 1 if "Order Rejected" in x else 0
    )

    df_flow["Compliant"] = df_flow["Steps"].apply(
        lambda x: 1 if x == process_flow else 0
    )
    df_flow["Compliant"] = np.where(df_flow["Rejected"] == 1, 1, df_flow["Compliant"])
    df_flow["Duplicated"] = df_flow["Duplicated_steps"].apply(
        lambda x: 1 if len(x) > 0 else 0
    )
    df_flow["Missing"] = df_flow["Missing_steps"].apply(
        lambda x: 1 if len(x) > 0 else 0
    )
    df_flow["Reversed"] = df_flow["Reversed_steps"].apply(
        lambda x: 1 if len(x) > 0 else 0
    )
    df_flow["Reversed_steps"] = df_flow["Reversed_steps"].map(
        remove_item_dico
    )  # remove the missing steps from the dico (already taken into account with the col Duplicated)

    # 0 instead of empty list or dict
    df_flow["Duplicated_steps"] = df_flow["Duplicated_steps"].apply(
        lambda x: 0 if len(x) == 0 else x
    )
    df_flow["Missing_steps"] = df_flow["Missing_steps"].apply(
        lambda x: 0 if len(x) == 0 else x
    )
    df_flow["Reversed_steps"] = df_flow["Reversed_steps"].apply(
        lambda x: 0 if len(x) == 0 else x
    )

    # 0 for the process rejected
    df_flow["Missing_steps"] = np.where(
        df_flow["Rejected"] == 1, 0, df_flow["Missing_steps"]
    )
    df_flow["Reversed_steps"] = np.where(
        df_flow["Rejected"] == 1, 0, df_flow["Reversed_steps"]
    )

    return df_flow


def most_skipped_steps(df):
    """Returns a dictionnary that lists the steps that were skipped and the number of variation of the process it happened
    Returns:
        dict: {step: nb of variations where the step was missed}
    """

    skipped_steps = list(df[df["Missing_steps"] != 0]["Missing_steps"].values)
    skipped_steps = [item for items in skipped_steps for item in items]
    mst_skppd_stps = dict(Counter(skipped_steps))
    return mst_skppd_stps


def most_duplicated_steps(df):
    """Returns a dictionnary that lists the steps that were duplicated and the number of variation of the process it happened
    Returns:
        dict: {step: nb of variations where the step was done more than once}
    """

    dup_steps = list(df[df["Duplicated_steps"] != 0]["Duplicated_steps"].values)
    dup_steps = [item for items in dup_steps for item in items]
    mst_dup_stps = dict(Counter(dup_steps))
    return mst_dup_stps


def most_reversed_steps(df, process_flow):
    """Returns a dictionary that list the steps of the process that were not made in the right order, and the number of times it happened
    Returns:
        dict: {step: nb of variations where the step was reversed}
    """

    reversed_steps = list(df[df["Reversed_steps"] != 0]["Reversed_steps"].values)

    dico_root = defaultdict(int)
    for element in reversed_steps:
        root_list = list(element["values_changed"].keys())
        for root in root_list:
            dico_root[root] += 1

    dico_root = dict(dico_root)

    dico_final = {}
    for key in dico_root.keys():
        a = int(re.findall(r"\d+", key)[0])
        new_name = process_flow[a]
        dico_final[new_name] = dico_root[key]

    sorted_dict = dict(
        sorted(dico_final.items(), key=operator.itemgetter(1), reverse=True)
    )
    return sorted_dict
