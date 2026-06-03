import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt

def evaluate(model, tasks, epochs):
    sklearn.set_config(enable_metadata_routing=True)

    cls = linear_model.LogisticRegression(
        solver="sag", n_jobs=-1, class_weight="balanced", max_iter=200
    )

    model.evaluate(
        ml_model_class=cls,
        ml_model_regression=linear_model.LinearRegression(),
        params=tasks,
        metric_class='roc_auc_ovo',
        metric_regression='r2',
        reference_methods=[],
        split_type="use-split",
        n_downsample=10000
    )
    plt.close("all")

    valid_scores = model.result.embedding_evaluation.loc[
        model.result.embedding_evaluation.score_split == "valid"
    ]

    avg_mltask_performance = valid_scores["value"].mean()
    # Per-task scores: {task_name: score}
    per_task_performance = (
        valid_scores.set_index("CLINIC_PARAM")["value"]
        .to_dict()
    )

    valid_recon_loss = float(model.result.sub_losses.get("recon_loss").get(epoch=-1, split="valid"))
    loss_per_epoch = {
        epoch: model.result.sub_losses.get("recon_loss").get(epoch=epoch, split="valid")
        for epoch in range(0, epochs)
    }

    return avg_mltask_performance, per_task_performance, valid_recon_loss, loss_per_epoch
