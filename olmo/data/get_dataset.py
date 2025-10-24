from olmo.data.academic_datasets import (
    ScienceQAImageOnly, OkVqa,
    TabWMPDirectAnswer,
    AndroidControl, AI2D, CountBenchQa, RealWorldQa, MathVista, MMMU, ClockBench,
    MuirBench
)
from olmo.data.academic_datasets_manual import (
    ChartQa, InfoQa, SceneTextQa, DocQa,
    TextVqa, AOkVqa, Vqa2, PlotQa, TallyQa, FigureQa, DvQa,
)
from olmo.data.dataset import Dataset
from olmo.data.pixmo_datasets import (
    PixMoDocs, PixMoCount, PixMoPoints, PixMoCapQa, PixMoCap, PixMoPointExplanations,
    PixMoAskModelAnything, PixMoPointsEval, DenseCaptionEval, PixMoClocks,
    CoSyn, CoSynPoint, CorrectionQa
)
import itertools
import os

from olmo.data.robot_datasets import *
from olmo.data.lvis_dataset import LVIS
from olmo.data.custom_lerobot_dataset import CustomLeRobotDataset


def get_dataset_by_name(dataset_name, split) -> Dataset:
    if dataset_name in ["scifi_document_qa", "pixmo_docs_other"]:
        return PixMoDocs("other", split=split)
    elif dataset_name in ["scifi_table_qa", "pixmo_docs_tables"]:
        return PixMoDocs("tables", split=split)
    elif dataset_name in ["scifi_diagram_qa", "pixmo_docs_diagrams"]:
        return PixMoDocs("diagrams", split=split)
    elif dataset_name in ["scifi_charts_qa", "pixmo_docs_charts"]:
        return PixMoDocs("charts", split=split)

    elif dataset_name in ["pixmo_docs_other_flat"]:
        return PixMoDocs("other", split=split, flat=True)
    elif dataset_name in ["pixmo_docs_charts_flat"]:
        return PixMoDocs("charts", split=split, flat=True)
    elif dataset_name in ["pixmo_docs_tables_flat"]:
        return PixMoDocs("tables", split=split, flat=True)
    elif dataset_name in ["pixmo_docs_diagrams_flat"]:
        return PixMoDocs("diagrams", split=split, flat=True)

    # CoSyn-400K / CoSyn-point
    doc_types = [
        "chart", "chemical", "circuit", "diagram",
        "document", "graphic", "math", "music",
        "nutrition", "table"
    ]
    cosyn_dataset_names = [f"cosyn_{doc_type}{suffix}" for doc_type, suffix in itertools.product(doc_types, ["", "_exp"])]
    if dataset_name == "cosyn_point":
        return CoSynPoint(split=split)
    elif dataset_name in cosyn_dataset_names:
        doc_type = dataset_name.split("_")[1]
        return CoSyn(doc_type, split=split, use_exp=dataset_name.endswith("_exp"))

    # PixMo-Pointing
    elif dataset_name in ["pointing_high_freq", "pixmo_points_high_freq"]:
        return PixMoPoints(kind="high_frequency", split=split, counting=False)
    elif dataset_name in ["point_count_high_freq", "pixmo_points_high_freq_counting"]:
        return PixMoPoints(kind="high_frequency", split=split, counting=True)
    elif dataset_name in ["pointing", "pixmo_points"]:
        return PixMoPoints(kind="basic", split=split, counting=False)
        # return PixMoPoints(kind="basic", split=split, counting=False, max_points=1, max_total_points_per_example=1)
    elif dataset_name in ["point_count", "pixmo_points_counting"]:
        return PixMoPoints(kind="basic", split=split, counting=True)

    # More than 60 points will start getting truncated anyway with a seq. len of 2304
    elif dataset_name in ["pixmo_points_train"]:
        return PixMoPoints(kind="basic", split=split, counting="both", max_points=60, max_total_points_per_example=60)
        # return PixMoPoints(kind="basic", split=split, counting=False, max_points=1, max_total_points_per_example=1, max_samples_per_image=1)
    elif dataset_name in ["pixmo_points_high_freq_train"]:
        return PixMoPoints(kind="high_frequency", split=split, counting="both", max_points=60, max_total_points_per_example=60)
    elif dataset_name in ["pixmo_count_train"]:
        return PixMoCount(split=split, counting="both")

    # PixMo-Point-Explanations
    elif dataset_name in ["point_qa", "pixmo_pointing_explanations"]:
        return PixMoPointExplanations(split=split, split_groups=True)

    # PixMo-Count
    elif dataset_name in ["fast_flickr_count_qa_point_count", "pixmo_count_counting"]:
        return PixMoCount(split=split, counting=True)
    elif dataset_name in ["fast_flickr_count_qa_pointing", "pixmo_count"]:
        return PixMoCount(split=split, counting=False)

    # PixMo-AskModelAnything
    elif dataset_name in ["user_qa", "pixmo_ask_model_anything"]:
        return PixMoAskModelAnything(split=split)

    # PixMo-CapQa
    elif dataset_name in ["synthetic_qa_v3", "pixmo_cap_qa"]:
        return PixMoCapQa(split=split)
    elif dataset_name in ["synthetic_qa_v3_as_user_qa", "pixmo_cap_qa_as_user_qa"]:
        return PixMoCapQa(split=split, style="user_qa")

    # PixMo-Cap
    if dataset_name in ["cockatoo_and_transcript_712k_sept6", "pixmo_cap_with_transcripts"]:
        return PixMoCap(split, mode="transcript_and_caption")
    if dataset_name in ["cockatoo_712k_sept6", "pixmo_cap"]:
        return PixMoCap(split, mode="captions")
    if dataset_name in ["pixmo_cap_transcript", "pixmo_transcript"]:
        return PixMoCap(split, mode="transcript")
    # if dataset_name in ["cockatoo_712k_sept6", "pixmo_cap"]:
    #     return PixMoCap(split, mode="captions")
    # if dataset_name in ["pixmo_transcript"]:
    #     return PixMoCap(split, mode="transcript")

    elif dataset_name in ["pixmo_clocks"]:
        return PixMoClocks(split=split)

    if dataset_name == "pointing_eval":
        assert split == "test"
        return PixMoPointsEval()

    # Multi-image Qa
    if dataset_name == "correction_qa":
        return CorrectionQa(split=split)
    elif dataset_name == "correction_qa_multi_only":
        return CorrectionQa(split=split, multi_image_only=True)
    # Filter out the qa pairs that contain more than 10 images
    elif dataset_name == "correction_qa_train":
        return CorrectionQa(split=split, max_images=10)
    elif dataset_name == "correction_qa_multi_only_train":
        return CorrectionQa(split=split, multi_image_only=True, max_images=10)

    # Academic datasets
    if dataset_name == "android_control":
        return AndroidControl(split)
    if dataset_name == "android_control_ll":
        return AndroidControl(split, mode="ll")
    if dataset_name == "chart_qa":
        return ChartQa(split, weighted=False)
    if dataset_name == "chart_qa_exp":
        return ChartQa(split, weighted=False, use_exp=True)
    if dataset_name == "real_world_qa_no_instruction":
        assert split == "test"
        return RealWorldQa("no_instruction")
    if dataset_name == "chart_qa_weighted":
        return ChartQa(split, weighted=True)
    if dataset_name == "info_qa":
        return InfoQa(split)
    if dataset_name == "doc_qa":
        return DocQa(split)
    if dataset_name == "science_qa_img":
        return ScienceQAImageOnly(split)
    if dataset_name == "coco_2014_vqa_multi":
        return Vqa2(split, multi_question=True)
    if dataset_name == "coco_2014_vqa":
        return Vqa2(split, multi_question=False)
    if dataset_name == "text_vqa":
        return TextVqa(split)
    if dataset_name == "plot_qa":
        return PlotQa(split)
    if dataset_name == "figure_qa":
        return FigureQa(dict(train="train", validation="validation1")[split])
    if dataset_name == "dv_qa":
        return DvQa(split)
    if dataset_name == "okvqa":
        return OkVqa(split)
    if dataset_name in ["mmmu"]:
        return MMMU(split)
    if dataset_name in ["mmmu_test"]:
        return MMMU(split)
    if dataset_name == "a_okvqa_da":
        return AOkVqa(split=split, direct_answer=True)
    if dataset_name == "a_okvqa_mc":
        return AOkVqa(split=split, direct_answer=False)
    if dataset_name == "st_qa":
        return SceneTextQa(split=split)
    if dataset_name == "tabwmp_da":
        return TabWMPDirectAnswer(split=split, include_options=False)
    if dataset_name == "countbench_qa":
        assert split == "huggingface"
        return CountBenchQa()
    if dataset_name == "tally_qa":
        return TallyQa(split=split)
    if dataset_name == "ai2_diagram_v2_mix_transparent":
        return AI2D(split=split, boxes="both")
    if dataset_name == "clock_bench":
        return ClockBench(split=split)
    if dataset_name == "dense_caption_eval":
        assert split == "test"
        return DenseCaptionEval()
    elif dataset_name == "math_vista_v2":
        if split == "validation":
            split = "testmini"
        return MathVista(split)
    if dataset_name == "muir_bench":
        return MuirBench(split)
    elif dataset_name == "muir_bench_mc":
        return MuirBench(split, use_mc_style=True)

    # pre-training
    if "bc_z" in dataset_name:
        return BC_Z(split="train", style="demo")
    if "bridge_data_v2" in dataset_name:
        return BridgeDataV2(split="train", style="demo")
    if "rt_1" in dataset_name:
        return RT_1(split="train", style="demo")
    if "auxiliary_depth_data" in dataset_name:
        return AuxiliaryDepthData(split="train", style="demo")
    if "auxiliary_trace_data" in dataset_name:
        return AuxiliaryTraceData(split="train", style="demo")
    if "lvis" in dataset_name:
        return LVIS(split="train", style="demo")
    
    # mid-training
    if "molmoact_dataset_home_primary" in dataset_name:
        return MolmoActDatasetHomePrimary(split="train", style="demo", width=320, height=240)
    if "molmoact_dataset_home_secondary" in dataset_name:
        return MolmoActDatasetHomeSecondary(split="train", style="demo", width=320, height=240)
    if "molmoact_dataset_tabletop_primary" in dataset_name:
        return MolmoActDatasetTabletopPrimary(split="train", style="demo", width=320, height=240)
    if "molmoact_dataset_tabletop_secondary" in dataset_name:
        return MolmoActDatasetTabletopSecondary(split="train", style="demo", width=320, height=240)

    # libero post-training
    if "libero_spatial" in dataset_name:
        return LIBEROSpatial(split="train", style="demo")
    if "libero_object" in dataset_name:
        return LIBEROObject(split="train", style="demo")
    if "libero_goal" in dataset_name:
        return LIBEROGoal(split="train", style="demo")
    if "libero_long" in dataset_name:
        return LIBEROLong(split="train", style="demo")
    
    # finetune
    if "finetune:" in dataset_name:
        path = dataset_name[len("finetune:"):]
        return CustomLeRobotDataset(path=path, high_res=False, style="demo")


    raise NotImplementedError(dataset_name, split)