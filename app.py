"""Aplicação Streamlit para o experimento de classificação de gatos e cachorros."""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Image as ReportImage,
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_SIZE = (64, 64)
BATCH_SIZE = 16
SEED = 42
CLASS_NAMES = ("cachorro", "gato")
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAINING_ZIP = PROJECT_DIR / "dataset_gato_cachorro.zip"
DEFAULT_CUSTOM_ZIP = PROJECT_DIR / "minha_base.zip"
RESULTS_CSV = PROJECT_DIR / "resultados_experimento.csv"
HISTORIES_JSON = PROJECT_DIR / "historias_experimento.json"

st.set_page_config(page_title="CNN Gato ou Cachorro", page_icon="🐱", layout="wide")


def safe_extract(uploaded_file: Any, destination: Path) -> Path:
    """Extract a ZIP and return the folder that contains the expected structure."""
    destination.mkdir(parents=True, exist_ok=True)
    if hasattr(uploaded_file, "getvalue"):
        zip_bytes = uploaded_file.getvalue()
    elif isinstance(uploaded_file, Path):
        zip_bytes = uploaded_file.read_bytes()
    else:
        zip_bytes = uploaded_file
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        members = archive.infolist()
        for member in members:
            target = (destination / member.filename).resolve()
            destination_root = str(destination.resolve()) + os.sep
            if not str(target).startswith(destination_root):
                raise ValueError("O arquivo ZIP contém um caminho inválido.")
        archive.extractall(destination)

    for candidate in [destination, *destination.rglob("*")]:
        if not candidate.is_dir():
            continue
        if all(
            (candidate / name).is_dir()
            for name in ("training_set", "test_set")
        ) or all((candidate / name).is_dir() for name in CLASS_NAMES):
            return candidate
    raise ValueError(
        "Não encontrei uma estrutura válida: use training_set/test_set "
        "ou as pastas cachorro/gato."
    )


def validate_training_dataset(dataset_dir: Path) -> None:
    """Validate the supplied dataset used for training."""
    required = ("training_set", "test_set")
    if not all((dataset_dir / name).is_dir() for name in required):
        raise ValueError(
            "A base de treino deve conter as pastas training_set e test_set."
        )
    for split in required:
        if not all((dataset_dir / split / name).is_dir() for name in CLASS_NAMES):
            raise ValueError(
                f"{split} deve conter exatamente as pastas cachorro e gato."
            )
    expected_counts = {"training_set": 2000, "test_set": 500}
    for split, expected in expected_counts.items():
        for class_name in CLASS_NAMES:
            count = sum(
                1
                for path in (dataset_dir / split / class_name).iterdir()
                if path.is_file()
                and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if count != expected:
                raise ValueError(
                    f"{split}/{class_name} deve conter {expected} imagens "
                    f"JPG ou PNG; encontrei {count}."
                )


def validate_custom_dataset(dataset_dir: Path) -> None:
    """Validate the 20-image dataset used only for final evaluation."""
    if not all((dataset_dir / name).is_dir() for name in CLASS_NAMES):
        raise ValueError("A base própria deve conter as pastas cachorro e gato.")
    counts = {}
    for class_name in CLASS_NAMES:
        counts[class_name] = sum(
            1
            for path in (dataset_dir / class_name).iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    if counts != {"cachorro": 10, "gato": 10}:
        raise ValueError(
            "A base própria deve conter exatamente 10 imagens JPG ou PNG "
            "em cada classe."
        )


def create_model() -> Sequential:
    model = Sequential(name="cnn_gato_cachorro")
    model.add(Input(shape=(*IMAGE_SIZE, 3)))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


def make_generators(dataset_dir: Path):
    validate_training_dataset(dataset_dir)
    train_dir = dataset_dir / "training_set"
    test_dir = dataset_dir / "test_set"
    validation_dir = dataset_dir / "validation_set"

    has_validation = validation_dir.is_dir()
    validation_split = 0.0 if has_validation else 0.2
    train_generator = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=7,
        horizontal_flip=True,
        shear_range=0.2,
        height_shift_range=0.07,
        zoom_range=0.2,
        validation_split=validation_split,
    )
    evaluation_generator = ImageDataGenerator(
        rescale=1 / 255, validation_split=validation_split
    )
    common = dict(
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )
    if has_validation:
        train = train_generator.flow_from_directory(
            train_dir, seed=SEED, shuffle=True, **common
        )
        validation = evaluation_generator.flow_from_directory(
            validation_dir, shuffle=False, **common
        )
    else:
        train = train_generator.flow_from_directory(
            train_dir, subset="training", seed=SEED, shuffle=True, **common
        )
        validation = evaluation_generator.flow_from_directory(
            train_dir, subset="validation", shuffle=False, **common
        )
    test = evaluation_generator.flow_from_directory(
        test_dir, shuffle=False, **common
    )
    return train, validation, test


def evaluate(model: Sequential, generator) -> dict[str, Any]:
    generator.reset()
    probabilities = model.predict(generator, verbose=0).ravel()
    predictions = (probabilities > 0.5).astype(int)
    actual = generator.classes
    errors = [
        {
            "file": generator.filepaths[index],
            "real": CLASS_NAMES[actual[index]],
            "previsto": CLASS_NAMES[predictions[index]],
            "confianca": float(
                probabilities[index]
                if predictions[index] == 1
                else 1 - probabilities[index]
            ),
        }
        for index in range(len(actual))
        if predictions[index] != actual[index]
    ]
    return {
        "accuracy": accuracy_score(actual, predictions),
        "precision": precision_score(actual, predictions, zero_division=0),
        "recall": recall_score(actual, predictions, zero_division=0),
        "f1": f1_score(actual, predictions, zero_division=0),
        "matrix": confusion_matrix(actual, predictions),
        "probabilities": probabilities,
        "predictions": predictions,
        "errors": errors,
    }


def train_model(dataset_dir: Path, epochs: int, learning_rate: float):
    tf.keras.utils.set_random_seed(SEED)
    train, validation, test = make_generators(dataset_dir)
    model = create_model()
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(train, epochs=epochs, validation_data=validation, verbose=0)
    return model, history, test, evaluate(model, test)


def save_experiment_artifacts(
    grid: pd.DataFrame,
    histories: dict[str, Any],
    model: Sequential,
    selected: dict[str, Any],
) -> None:
    grid.to_csv(RESULTS_CSV, index=False)
    model.save(
        PROJECT_DIR
        / f"modelo_e{int(selected['epocas'])}_lr{selected['learning_rate']:g}.keras"
    )
    saved_histories = {
        key: {
            metric: [float(value) for value in values]
            for metric, values in history.history.items()
        }
        for key, history in histories.items()
    }
    HISTORIES_JSON.write_text(
        json.dumps(saved_histories, ensure_ascii=False),
        encoding="utf-8",
    )


def load_saved_experiment(dataset_dir: Path) -> dict[str, Any] | None:
    if not RESULTS_CSV.is_file() or not HISTORIES_JSON.is_file():
        return None

    grid = pd.read_csv(RESULTS_CSV)
    expected_configs = {
        (epochs, learning_rate)
        for learning_rate in (0.1, 0.01, 0.001)
        for epochs in (10, 15, 20)
    }
    if len(grid) != 9 or set(zip(grid["epocas"], grid["learning_rate"])) != expected_configs:
        raise ValueError("O CSV salvo não contém exatamente as 9 configurações exigidas.")

    grid = grid.sort_values(
        ["learning_rate", "epocas"], ascending=[False, True]
    ).reset_index(drop=True)
    if "id" not in grid:
        grid.insert(0, "id", [f"E{index}" for index in range(1, 10)])
    if "pct_gato_previsto" not in grid:
        grid["pct_gato_previsto"] = [
            min(1.0, float(row["recall_teste"]) * 0.5 / float(row["precisao_teste"]))
            if row["precisao_teste"] > 0
            else 0.0
            for _, row in grid.iterrows()
        ]

    selected = grid.sort_values(
        ["acc_validacao", "perda_validacao"], ascending=[False, True]
    ).iloc[0].to_dict()
    model_path = (
        PROJECT_DIR
        / f"modelo_e{int(selected['epocas'])}_lr{selected['learning_rate']:g}.keras"
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"Arquivo do modelo selecionado não encontrado: {model_path}")

    with HISTORIES_JSON.open(encoding="utf-8") as file:
        stored_histories = json.load(file)
    if set(stored_histories) != {f"E{index}" for index in range(1, 10)}:
        raise ValueError("O arquivo de curvas não contém as 9 configurações.")
    histories = {
        key: SimpleNamespace(history=history)
        for key, history in stored_histories.items()
    }
    model = tf.keras.models.load_model(model_path)
    _, _, test_generator = make_generators(dataset_dir)
    test_result = evaluate(model, test_generator)
    return {
        "grid": grid,
        "selected_config": selected,
        "histories": histories,
        "model": model,
        "history": histories[selected["id"]],
        "test_result": test_result,
    }


def render_metrics(result: dict[str, Any], title: str) -> None:
    st.subheader(title)
    values = [
        ("Acurácia", result["accuracy"]),
        ("Precisão", result["precision"]),
        ("Recall", result["recall"]),
        ("F1-score", result["f1"]),
    ]
    columns = st.columns(4)
    for column, (label, value) in zip(columns, values):
        column.metric(label, f"{value:.2%}")
    matrix = pd.DataFrame(
        result["matrix"],
        index=["real: cachorro", "real: gato"],
        columns=["previsto: cachorro", "previsto: gato"],
    )
    st.dataframe(matrix, use_container_width=True)


def history_chart(history) -> None:
    chart = pd.DataFrame(
        {
            "Treinamento": history.history["accuracy"],
            "Validação": history.history["val_accuracy"],
        },
        index=range(1, len(history.history["accuracy"]) + 1),
    )
    st.line_chart(chart)


def history_drawing(history: Any, title: str) -> Drawing:
    drawing = Drawing(500, 220)
    chart = LinePlot()
    epochs = range(1, len(history.history["accuracy"]) + 1)
    chart.data = [
        list(zip(epochs, history.history["accuracy"])),
        list(zip(epochs, history.history["val_accuracy"])),
        list(zip(epochs, history.history["loss"])),
        list(zip(epochs, history.history["val_loss"])),
    ]
    chart.x = 45
    chart.y = 35
    chart.width = 420
    chart.height = 150
    chart.xValueAxis.valueMin = 1
    chart.xValueAxis.valueMax = len(history.history["accuracy"])
    chart.yValueAxis.valueMin = 0
    chart.yValueAxis.valueMax = max(
        max(history.history["accuracy"]),
        max(history.history["val_accuracy"]),
        max(history.history["loss"]),
        max(history.history["val_loss"]),
        1,
    )
    chart.lines[0].strokeColor = colors.HexColor("#4C78A8")
    chart.lines[1].strokeColor = colors.HexColor("#F58518")
    chart.lines[2].strokeColor = colors.HexColor("#54A24B")
    chart.lines[3].strokeColor = colors.HexColor("#E45756")
    drawing.add(String(45, 195, title))
    drawing.add(chart)
    drawing.add(String(45, 15, "azul/laranja: acuracia treino/validacao; verde/vermelho: perda"))
    return drawing


def serialize_model(model: Sequential) -> bytes:
    descriptor, filename = tempfile.mkstemp(suffix=".keras")
    os.close(descriptor)
    path = Path(filename)
    try:
        model.save(path)
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def build_discussion(
    grid: pd.DataFrame,
    selected: dict[str, Any],
    test_result: dict[str, Any],
    custom_result: dict[str, Any] | None,
    observation: str,
    improvement: str,
) -> list[str]:
    answers = []
    learning_rate_summaries = []
    for learning_rate in (0.1, 0.01, 0.001):
        subset = grid[grid["learning_rate"] == learning_rate]
        mean_train = subset["acc_treino"].mean()
        mean_validation = subset["acc_validacao"].mean()
        positive_ratio = subset["pct_gato_previsto"].mean()
        collapsed_configs = subset[
            (subset["pct_gato_previsto"] <= 0.1)
            | (subset["pct_gato_previsto"] >= 0.9)
        ]["id"].tolist()
        collapse_text = (
            "houve previsao quase exclusiva de uma classe em "
            + ", ".join(collapsed_configs)
            if collapsed_configs
            else "nao houve previsao quase exclusiva de uma classe"
        )
        learning_rate_summaries.append(
            f"Learning rate {learning_rate:g}: acuracia media de treino "
            f"{mean_train:.2%}, validacao {mean_validation:.2%}; {collapse_text} "
            f"(fracao media de gatos previstos: {positive_ratio:.1%})."
        )
    answers.append(" ".join(learning_rate_summaries))

    epoch_summary = grid.groupby("epocas")[["acc_treino", "acc_validacao"]].mean()
    epoch_values = list(epoch_summary.index)
    train_increased = epoch_summary.iloc[-1]["acc_treino"] > epoch_summary.iloc[0]["acc_treino"]
    validation_improved = (
        epoch_summary.iloc[-1]["acc_validacao"]
        > epoch_summary.iloc[0]["acc_validacao"]
    )
    if train_increased and not validation_improved:
        epoch_analysis = (
            "A media de treino cresceu, mas a de validacao nao melhorou entre "
            f"{epoch_values[0]} e {epoch_values[-1]} epocas, o que e um sinal "
            "possivel de overfitting."
        )
    elif validation_improved:
        epoch_analysis = (
            f"A acuracia media de validacao mudou de "
            f"{epoch_summary.iloc[0]['acc_validacao']:.2%} para "
            f"{epoch_summary.iloc[-1]['acc_validacao']:.2%} entre "
            f"{epoch_values[0]} e {epoch_values[-1]} epocas; mais epocas "
            "nao garantem melhora em todas as configuracoes."
        )
    else:
        epoch_analysis = (
            f"A acuracia media de validacao nao melhorou entre "
            f"{epoch_values[0]} e {epoch_values[-1]} epocas."
        )
    answers.append(epoch_analysis)

    selected_row = grid[grid["id"] == selected["id"]].iloc[0]
    best_test = grid.loc[grid["acc_teste"].idxmax()]
    test_comparison = (
        "sim" if selected["id"] == best_test["id"] else "nao"
    )
    answers.append(
        f"A configuracao selecionada foi {selected['id']} ({int(selected['epocas'])} "
        f"epocas, learning rate {selected['learning_rate']:g}), por ter acuracia "
        f"de validacao {selected_row['acc_validacao']:.2%} e perda "
        f"{selected_row['perda_validacao']:.4f}. Ela tambem obteve a maior "
        f"acuracia de teste: {test_comparison}."
    )

    if custom_result is None:
        answers.append("A avaliacao da base propria ainda nao foi executada.")
    else:
        accuracy_delta = custom_result["accuracy"] - test_result["accuracy"]
        answers.append(
            f"Na base de teste, acuracia {test_result['accuracy']:.2%}; na base "
            f"propria, {custom_result['accuracy']:.2%} (diferenca "
            f"{accuracy_delta:+.2%}). A diferenca e compativel com possivel "
            "mudanca de distribuicao; interprete-a considerando que a base "
            "propria tem somente 20 imagens."
        )
    high_confidence_errors = (
        [
            f"{Path(error['file']).name} ({error['confianca']:.1%})"
            for error in custom_result["errors"]
            if error["confianca"] > 0.9
        ]
        if custom_result
        else []
    )
    answers.append(
        "Erros com confianca acima de 90%: "
        + (", ".join(escape(name) for name in high_confidence_errors) or "nenhum")
        + ". Observacao do estudante: "
        + (escape(observation.strip()) or "nenhuma observacao informada.")
    )
    answers.append(
        "Melhoria proposta: "
        + (escape(improvement.strip()) or "Nenhuma proposta informada.")
    )
    return answers


def generate_report_pdf() -> bytes:
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=landscape(A4), title="Relatorio CNN Gato ou Cachorro"
    )
    styles = getSampleStyleSheet()
    content = [
        Paragraph("Relatorio do Experimento - CNN Gato ou Cachorro", styles["Title"]),
        Spacer(1, 12),
        Paragraph("1. Introducao", styles["Heading2"]),
        Paragraph(
            "O experimento classifica imagens de gatos e cachorros com uma rede "
            "neural convolucional. O objetivo e comparar nove configuracoes e "
            "verificar a generalizacao em uma base propria de 20 imagens.",
            styles["BodyText"],
        ),
        Paragraph("2. Metodologia", styles["Heading2"]),
        Paragraph(
            "A aplicacao utilizou uma CNN com duas camadas convolucionais, "
            "pooling 2x2, duas camadas densas de 128 neuronios e saida sigmoid. "
            "As imagens foram normalizadas e redimensionadas para 64x64 pixels. "
            "A base fornecida foi usada somente em treino, validacao e teste; "
            "a base propria foi reservada para a avaliacao final.",
            styles["BodyText"],
        ),
        Paragraph(
            "Foram comparadas 10, 15 e 20 epocas com learning rates 0,1, 0,01 "
            "e 0,001. A selecao usa a maior acuracia de validacao e, em empate, "
            "a menor perda de validacao. Gato e a classe positiva.",
            styles["BodyText"],
        ),
    ]

    grid = st.session_state.get("grid")
    selected = None
    if grid is not None:
        content.append(Paragraph("3. Resultados: nove configuracoes", styles["Heading2"]))
        table_data = [[
            "ID", "Epocas", "Learning rate", "Acc. treino", "Acc. validacao",
            "Perda validacao", "Acc. teste", "Precisao teste", "Recall teste",
            "F1 teste",
        ]]
        for _, row in grid.iterrows():
            table_data.append([
                str(row["id"]),
                str(int(row["epocas"])),
                f"{row['learning_rate']:g}",
                f"{row['acc_treino']:.4f}",
                f"{row['acc_validacao']:.4f}",
                f"{row['perda_validacao']:.4f}",
                f"{row['acc_teste']:.4f}",
                f"{row['precisao_teste']:.4f}",
                f"{row['recall_teste']:.4f}",
                f"{row['f1_teste']:.4f}",
            ])
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C78A8")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
        ]))
        content.extend([table, Spacer(1, 12)])

        selected = st.session_state.get("selected_config")
        if selected:
            content.append(Paragraph(
                "Modelo selecionado pela maior acuracia de validacao "
                "(desempate pela menor perda de validacao): "
                f"{selected['id']}, {int(selected['epocas'])} epocas, learning rate "
                f"{selected['learning_rate']:g}.",
                styles["BodyText"],
            ))
        histories = st.session_state.get("histories", {})
        selected_history = histories.get(selected["id"]) if selected else None
        low_rate_history = next(
            (history for key, history in histories.items() if key.startswith("E1") or key.startswith("E2") or key.startswith("E3")),
            None,
        )
        if selected_history is not None:
            content.append(history_drawing(selected_history, "Curvas do modelo selecionado"))
        if low_rate_history is not None:
            content.append(history_drawing(low_rate_history, "Curvas de uma configuracao com learning rate 0,1"))

    test_result = st.session_state.get("test_result")
    if test_result:
        content.extend([
            Paragraph("Resultados na base de teste", styles["Heading3"]),
            Paragraph(
                f"Acuracia: {test_result['accuracy']:.2%}<br/>"
                f"Precisao: {test_result['precision']:.2%}<br/>"
                f"Recall: {test_result['recall']:.2%}<br/>"
                f"F1-score: {test_result['f1']:.2%}",
                styles["BodyText"],
            ),
            Paragraph(
                "Matriz de confusao:<br/>"
                f"{test_result['matrix'].tolist()}",
                styles["BodyText"],
            ),
        ])

    custom_result = st.session_state.get("custom_result")
    if test_result or custom_result:
        content.append(Paragraph("Tabela 2: comparacao das bases", styles["Heading3"]))
        comparison_data = [[
            "Base", "Acuracia", "Precisao", "Recall", "F1-score"
        ]]
        for label, result in (
            ("Teste", test_result),
            ("Minha base", custom_result),
        ):
            if result:
                comparison_data.append([
                    label,
                    f"{result['accuracy']:.4f}",
                    f"{result['precision']:.4f}",
                    f"{result['recall']:.4f}",
                    f"{result['f1']:.4f}",
                ])
        comparison_table = Table(comparison_data, repeatRows=1)
        comparison_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C78A8")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        content.append(comparison_table)

    if custom_result:
        content.extend([
            Paragraph("Resultados na minha base", styles["Heading3"]),
            Paragraph(
                f"Acuracia: {custom_result['accuracy']:.2%}<br/>"
                f"Precisao: {custom_result['precision']:.2%}<br/>"
                f"Recall: {custom_result['recall']:.2%}<br/>"
                f"F1-score: {custom_result['f1']:.2%}",
                styles["BodyText"],
            ),
            Paragraph(
                "Matriz de confusao (Tabela 3):<br/>"
                "Real cachorro: VN="
                f"{int(custom_result['matrix'][0][0])}, FP={int(custom_result['matrix'][0][1])}<br/>"
                "Real gato: FN="
                f"{int(custom_result['matrix'][1][0])}, VP={int(custom_result['matrix'][1][1])}",
                styles["BodyText"],
            ),
            Paragraph(
                "Erros observados:<br/>"
                + ("<br/>".join(
                    f"{escape(Path(error['file']).name)}: real={error['real']}, "
                    f"previsto={error['previsto']}, confiança={error['confianca']:.2%}"
                    for error in custom_result["errors"]
                ) or "Nenhum erro encontrado."),
                styles["BodyText"],
            ),
        ])

    if grid is not None and selected is not None and test_result is not None:
        selected_row = grid[grid["id"] == selected["id"]].iloc[0]
        content.append(Paragraph("4. Discussao", styles["Heading2"]))
        for index, answer in enumerate(
            build_discussion(
                grid,
                selected,
                test_result,
                custom_result,
                st.session_state.get("error_observation", ""),
                st.session_state.get("improvement_proposal", ""),
            ),
            start=1,
        ):
            content.append(
                Paragraph(f"{index}. {answer}", styles["BodyText"])
            )

        if custom_result and custom_result["errors"]:
            content.append(Paragraph("Exemplos de erros na base propria", styles["Heading3"]))
            for error in custom_result["errors"][:4]:
                image = ReportImage(error["file"], width=110, height=85, kind="proportional")
                content.append(
                    Table(
                        [[
                            image,
                            Paragraph(
                                f"{escape(Path(error['file']).name)}: real={error['real']}, "
                                f"previsto={error['previsto']}, "
                                f"confianca={error['confianca']:.2%}",
                                styles["BodyText"],
                            ),
                        ]],
                        colWidths=[125, 500],
                    )
                )

        content.extend([
            Paragraph("5. Conclusao", styles["Heading2"]),
            Paragraph(
                f"A melhor configuracao pela validacao foi {selected['id']} "
                f"({int(selected['epocas'])} epocas, learning rate "
                f"{selected['learning_rate']:g}), com acuracia de validacao "
                f"{selected_row['acc_validacao']:.2%}. "
                + (
                    f"A acuracia na base propria foi {custom_result['accuracy']:.2%}. "
                    if custom_result
                    else "A base propria ainda nao foi avaliada. "
                )
                + (escape(st.session_state.get("improvement_proposal", "").strip())
                   or "A melhoria proposta deve ser registrada pelo estudante."),
                styles["BodyText"],
            ),
        ])
    document.build(content)
    return buffer.getvalue()


def predict_image(model: Sequential, image: Image.Image) -> tuple[str, float]:
    rgb_image = image.convert("RGB").resize(IMAGE_SIZE)
    pixels = np.asarray(rgb_image, dtype=np.float32) / 255.0
    probability = float(model.predict(pixels[np.newaxis, ...], verbose=0).ravel()[0])
    label = "gato" if probability > 0.5 else "cachorro"
    confidence = probability if probability > 0.5 else 1 - probability
    return label, confidence


def show_dataset_preview(root: Path, limit: int | None = 5) -> None:
    st.subheader("Imagens da base")
    if not all((root / name).is_dir() for name in CLASS_NAMES):
        root = root / "training_set"
    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        images = [
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        st.write(f"**Classe: {class_name}** — {len(images)} imagens")
        visible_images = images if limit is None else images[:limit]
        for offset in range(0, len(visible_images), 5):
            columns = st.columns(min(5, len(visible_images) - offset))
            for column, image_path in zip(columns, visible_images[offset : offset + 5]):
                column.image(
                    str(image_path),
                    caption=image_path.name,
                    use_container_width=True,
                )


st.title("🐱 CNN — Gato ou Cachorro")
st.caption(
    "Aplicação do experimento do material: data augmentation, treino, validação, "
    "teste e avaliação da sua própria base."
)

with st.sidebar:
    st.header("Configuração")
    dataset_zip = st.file_uploader(
        "Base fornecida para treino (.zip)", type="zip", key="dataset"
    )
    epochs = st.select_slider("Épocas", options=[10, 15, 20], value=10)
    learning_rate = st.select_slider(
        "Taxa de aprendizado",
        options=[0.1, 0.01, 0.001],
        value=0.001,
        format_func=lambda value: f"{value:g}",
    )
    train_button = st.button("Treinar configuração", type="primary")
    grid_button = st.button("Executar grade com 9 configurações")

if "dataset_dir" not in st.session_state:
    st.session_state.dataset_dir = None
if "custom_dataset_dir" not in st.session_state:
    st.session_state.custom_dataset_dir = None

if st.session_state.custom_dataset_dir is None and DEFAULT_CUSTOM_ZIP.is_file():
    custom_folder = Path(tempfile.mkdtemp(prefix="cnn_custom_default_"))
    try:
        custom_dir = safe_extract(DEFAULT_CUSTOM_ZIP, custom_folder)
        validate_custom_dataset(custom_dir)
        st.session_state.custom_dataset_dir = str(custom_dir)
        st.session_state.custom_dataset_name = DEFAULT_CUSTOM_ZIP.name
    except (ValueError, zipfile.BadZipFile) as error:
        shutil.rmtree(custom_folder, ignore_errors=True)
        st.warning(f"Não foi possível carregar a base própria padrão: {error}")

if dataset_zip is None and st.session_state.dataset_dir is None:
    if DEFAULT_TRAINING_ZIP.is_file():
        folder = Path(tempfile.mkdtemp(prefix="cnn_dataset_"))
        try:
            st.session_state.dataset_dir = str(
                safe_extract(DEFAULT_TRAINING_ZIP, folder)
            )
            validate_training_dataset(Path(st.session_state.dataset_dir))
            st.session_state.dataset_name = DEFAULT_TRAINING_ZIP.name
            st.info("Usando automaticamente a base fornecida dataset_gato_cachorro.zip.")
        except (ValueError, zipfile.BadZipFile) as error:
            shutil.rmtree(folder, ignore_errors=True)
            st.error(str(error))

if dataset_zip is not None:
    should_load = st.session_state.get("dataset_name") != dataset_zip.name
    if should_load:
        if st.session_state.dataset_dir:
            shutil.rmtree(st.session_state.dataset_dir, ignore_errors=True)
        folder = Path(tempfile.mkdtemp(prefix="cnn_dataset_"))
        try:
            st.session_state.dataset_dir = str(safe_extract(dataset_zip, folder))
            validate_training_dataset(Path(st.session_state.dataset_dir))
            st.session_state.dataset_name = dataset_zip.name
            st.success("Dataset principal carregado.")
        except (ValueError, zipfile.BadZipFile) as error:
            shutil.rmtree(folder, ignore_errors=True)
            st.session_state.dataset_dir = None
            st.error(str(error))

if st.session_state.dataset_dir is None:
    st.info(
        "Envie o ZIP da base fornecida para treino. Ela deve conter "
        "training_set/cachorro, training_set/gato, test_set/cachorro e "
        "test_set/gato. A minha_base.zip deve ser usada somente na aba de "
        "avaliação final."
    )
    st.stop()

dataset_dir = Path(st.session_state.dataset_dir)
st.write(
    f"**Dataset:** `{st.session_state.get('dataset_name', 'dataset.zip')}` · "
    f"classes: `{', '.join(CLASS_NAMES)}` · imagem: `64 × 64 RGB`"
)
show_dataset_preview(dataset_dir)

if (
    "grid" not in st.session_state
    and st.session_state.get("dataset_name") == DEFAULT_TRAINING_ZIP.name
):
    saved_run = load_saved_experiment(dataset_dir)
    if saved_run:
        for key, value in saved_run.items():
            st.session_state[key] = value
        st.info("Resultados executados do notebook foram carregados automaticamente.")

if train_button:
    for key in ("grid", "histories", "selected_config", "custom_result"):
        st.session_state.pop(key, None)
    with st.spinner("Treinando a CNN..."):
        model, history, test_generator, result = train_model(
            dataset_dir, epochs, learning_rate
        )
    st.session_state.model = model
    st.session_state.test_result = result
    st.session_state.history = history
    st.session_state.test_generator = test_generator
    st.session_state.config = {"epocas": epochs, "learning_rate": learning_rate}

if grid_button:
    rows = []
    histories = {}
    best_model = None
    best_history = None
    best_result = None
    best_key = None
    progress = st.progress(0, text="Executando grade...")
    configurations = [(e, lr) for lr in (0.1, 0.01, 0.001) for e in (10, 15, 20)]
    for index, (grid_epochs, grid_lr) in enumerate(configurations, start=1):
        with st.spinner(f"Treinando {grid_epochs} épocas · lr {grid_lr:g}..."):
            model, grid_history, _, result = train_model(
                dataset_dir, grid_epochs, grid_lr
            )
        rows.append(
            {
                "id": f"E{index}",
                "epocas": grid_epochs,
                "learning_rate": grid_lr,
                "acc_treino": float(grid_history.history["accuracy"][-1]),
                "acc_validacao": float(grid_history.history["val_accuracy"][-1]),
                "perda_validacao": float(grid_history.history["val_loss"][-1]),
                "acc_teste": result["accuracy"],
                "precisao_teste": result["precision"],
                "recall_teste": result["recall"],
                "f1_teste": result["f1"],
                "pct_gato_previsto": float(result["predictions"].mean()),
            }
        )
        histories[f"E{index}"] = grid_history
        current_key = (rows[-1]["acc_validacao"], -rows[-1]["perda_validacao"])
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_model = model
            best_history = grid_history
            best_result = result
        progress.progress(index / len(configurations), text=f"{index}/9 concluídas")
    st.session_state.grid = pd.DataFrame(rows)
    st.session_state.histories = histories
    st.session_state.model = best_model
    st.session_state.history = best_history
    st.session_state.test_result = best_result
    st.session_state.selected_config = st.session_state.grid.sort_values(
        ["acc_validacao", "perda_validacao"], ascending=[False, True]
    ).iloc[0].to_dict()
    save_experiment_artifacts(
        st.session_state.grid,
        histories,
        best_model,
        st.session_state.selected_config,
    )

if "grid" in st.session_state:
    st.header("Experimento: 9 configurações")
    st.dataframe(st.session_state.grid, use_container_width=True)
    selected = st.session_state["selected_config"]
    st.success(
        "Modelo selecionado pela validação: "
        f"{int(selected['epocas'])} épocas · lr {selected['learning_rate']:g}"
    )
    st.download_button(
        "Baixar resultados_experimento.csv",
        st.session_state.grid.to_csv(index=False).encode("utf-8"),
        "resultados_experimento.csv",
        "text/csv",
    )

if "test_result" in st.session_state:
    render_metrics(st.session_state.test_result, "Resultado na base de teste")
    st.subheader("Curvas de aprendizado")
    history_chart(st.session_state.history)
    st.download_button(
        "Baixar modelo selecionado (.keras)",
        data=serialize_model(st.session_state.model),
        file_name="modelo_selecionado.keras",
        mime="application/octet-stream",
    )

st.header("Testar novas imagens")
st.write(
    "Aceita JPG, JPEG e PNG. Primeiro treine uma configuração; "
    "a avaliação pode ser feita por imagem isolada, câmera ou ZIP."
)

test_tab, camera_tab, zip_tab = st.tabs(
    ["Imagens soltas", "Tirar foto", "Base em ZIP"]
)

with test_tab:
    uploaded_images = st.file_uploader(
        "Escolha uma ou várias imagens",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="individual_images",
    )
    if uploaded_images:
        if "model" not in st.session_state:
            st.warning("Treine o modelo antes de testar imagens.")
        else:
            columns = st.columns(min(3, len(uploaded_images)))
            for index, uploaded in enumerate(uploaded_images):
                image = Image.open(uploaded)
                label, confidence = predict_image(st.session_state.model, image)
                column = columns[index % len(columns)]
                column.image(image, caption=uploaded.name, use_container_width=True)
                column.success(
                    f"{label.upper()} — confiança {confidence:.1%}"
                )

with camera_tab:
    camera_image = st.camera_input("Tire uma foto")
    if camera_image:
        if "model" not in st.session_state:
            st.warning("Treine o modelo antes de testar a foto.")
        else:
            image = Image.open(camera_image)
            label, confidence = predict_image(st.session_state.model, image)
            st.image(image, caption="Foto tirada pela câmera", use_container_width=True)
            st.success(f"Resultado: **{label.upper()}**")
            st.metric("Confiança", f"{confidence:.1%}")

with zip_tab:
    st.write(
        "Envie `minha_base.zip` contendo exatamente 10 imagens em cada uma das "
        "pastas `cachorro` e `gato`. A base incluída no projeto já foi carregada "
        "automaticamente e não participa do treinamento."
    )
    custom_zip = st.file_uploader("Minha base (.zip)", type="zip", key="custom")
    if custom_zip is not None:
        custom_root = Path(tempfile.mkdtemp(prefix="cnn_custom_"))
        try:
            custom_dir = safe_extract(custom_zip, custom_root)
            validate_custom_dataset(custom_dir)
            st.session_state.custom_dataset_dir = str(custom_dir)
            st.session_state.custom_dataset_name = custom_zip.name
        except (ValueError, zipfile.BadZipFile) as error:
            st.error(str(error))
            shutil.rmtree(custom_root, ignore_errors=True)

    custom_dir = st.session_state.get("custom_dataset_dir")
    if custom_dir:
        st.caption(
            f"Base própria carregada: {st.session_state.get('custom_dataset_name', 'minha_base.zip')}"
        )
        show_dataset_preview(Path(custom_dir), limit=None)
        if "model" in st.session_state:
            generator = ImageDataGenerator(rescale=1 / 255).flow_from_directory(
                custom_dir,
                target_size=IMAGE_SIZE,
                batch_size=BATCH_SIZE,
                class_mode="binary",
                shuffle=False,
            )
            custom_result = evaluate(st.session_state.model, generator)
            st.session_state.custom_result = custom_result
            render_metrics(custom_result, "Resultado na minha base")
            st.write(
                "A classe positiva é `gato` (mapeamento alfabético: "
                "`cachorro → 0`, `gato → 1`)."
            )
            if custom_result["errors"]:
                st.subheader("Imagens classificadas incorretamente")
                st.dataframe(
                    pd.DataFrame(custom_result["errors"]),
                    use_container_width=True,
                )
                error_columns = st.columns(min(4, len(custom_result["errors"])))
                for column, error in zip(error_columns, custom_result["errors"][:4]):
                    column.image(
                        error["file"],
                        caption=(
                            f"Real: {error['real']} · previsto: {error['previsto']} · "
                            f"confiança: {error['confianca']:.1%}"
                        ),
                        use_container_width=True,
                    )

        st.subheader("Anotações para a discussão do relatório")
        st.session_state.error_observation = st.text_area(
            "Que padrão observou nos erros? Considere iluminação, distância, "
            "ângulo, fundo, raça ou confiança da previsão.",
            value=st.session_state.get("error_observation", ""),
            key="error_observation_input",
        )
        st.session_state.improvement_proposal = st.text_area(
            "Que melhoria você proporia para o modelo e por quê?",
            value=st.session_state.get("improvement_proposal", ""),
            key="improvement_proposal_input",
        )

st.divider()
st.header("Relatorio")
if (
    "grid" in st.session_state
    and "test_result" in st.session_state
    and "custom_result" in st.session_state
):
    pdf_report = generate_report_pdf()
    st.download_button(
        "Gerar e baixar relatorio PDF",
        data=pdf_report,
        file_name="relatorio_experimento_cnn.pdf",
        mime="application/pdf",
        type="primary",
    )
else:
    st.info(
        "Execute a grade com 9 configurações e avalie a base própria antes "
        "de gerar o relatório completo."
    )
