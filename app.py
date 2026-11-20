"""Aplicação Streamlit para o experimento de classificação de gatos e cachorros."""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import zipfile
import random
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
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
DEFAULT_DATASET_ZIP = PROJECT_DIR / "minha_base.zip"

st.set_page_config(page_title="CNN Gato ou Cachorro", page_icon="🐱", layout="wide")


def safe_extract(
    uploaded_file: Any, destination: Path, *, dataset_zip: bool = False
) -> Path:
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
    train_dir = dataset_dir / "training_set"
    test_dir = dataset_dir / "test_set"
    validation_dir = dataset_dir / "validation_set"
    if not train_dir.is_dir() or not test_dir.is_dir():
        if all((dataset_dir / name).is_dir() for name in CLASS_NAMES):
            split_dir = Path(tempfile.mkdtemp(prefix="cnn_20_split_"))
            train_dir = split_dir / "training_set"
            validation_dir = split_dir / "validation_set"
            test_dir = split_dir / "test_set"
            random.seed(SEED)
            for class_name in CLASS_NAMES:
                images = [
                    path
                    for path in (dataset_dir / class_name).iterdir()
                    if path.is_file()
                    and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ]
                random.shuffle(images)
                if len(images) < 5:
                    raise ValueError(
                        "A base própria precisa de pelo menos 5 imagens por classe."
                    )
                train_count = max(1, int(len(images) * 0.6))
                validation_count = max(1, int(len(images) * 0.2))
                test_count = len(images) - train_count - validation_count
                groups = (
                    (train_dir, images[:train_count]),
                    (
                        validation_dir,
                        images[train_count : train_count + validation_count],
                    ),
                    (test_dir, images[train_count + validation_count :]),
                )
                for split_folder, split_images in groups:
                    class_folder = split_folder / class_name
                    class_folder.mkdir(parents=True, exist_ok=True)
                    for image in split_images:
                        shutil.copy2(image, class_folder / image.name)
        else:
            raise ValueError("O dataset precisa conter training_set/test_set ou cachorro/gato.")

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
    return {
        "accuracy": accuracy_score(actual, predictions),
        "precision": precision_score(actual, predictions, zero_division=0),
        "recall": recall_score(actual, predictions, zero_division=0),
        "f1": f1_score(actual, predictions, zero_division=0),
        "matrix": confusion_matrix(actual, predictions),
        "probabilities": probabilities,
        "predictions": predictions,
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


def generate_report_pdf() -> bytes:
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, title="Relatorio CNN Gato ou Cachorro")
    styles = getSampleStyleSheet()
    content = [
        Paragraph("Relatorio do Experimento - CNN Gato ou Cachorro", styles["Title"]),
        Spacer(1, 12),
        Paragraph("1. Metodologia", styles["Heading2"]),
        Paragraph(
            "A aplicacao utilizou uma CNN com duas camadas convolucionais, "
            "pooling 2x2, duas camadas densas de 128 neuronios e saida sigmoid. "
            "As imagens foram normalizadas e redimensionadas para 64x64 pixels.",
            styles["BodyText"],
        ),
        Paragraph("A avaliacao considera gato como a classe positiva.", styles["BodyText"]),
    ]

    grid = st.session_state.get("grid")
    if grid is not None:
        content.append(Paragraph("2. Comparacao das 9 configuracoes", styles["Heading2"]))
        table_data = [["Epocas", "Learning rate", "Acc. validacao", "Perda validacao", "Acc. teste"]]
        for _, row in grid.iterrows():
            table_data.append([
                str(int(row["epocas"])),
                f"{row['learning_rate']:g}",
                f"{row['acc_validacao']:.2%}",
                f"{row['perda_validacao']:.4f}",
                f"{row['acc_teste']:.2%}",
            ])
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C78A8")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        content.extend([table, Spacer(1, 12)])

        selected = st.session_state.get("selected_config")
        if selected:
            content.append(Paragraph(
                "Modelo selecionado pela maior acuracia de validacao "
                "(desempate pela menor perda de validacao): "
                f"{int(selected['epocas'])} epocas, learning rate "
                f"{selected['learning_rate']:g}.",
                styles["BodyText"],
            ))

    test_result = st.session_state.get("test_result")
    if test_result:
        content.extend([
            Paragraph("3. Resultados na base de teste", styles["Heading2"]),
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
    if custom_result:
        content.extend([
            Paragraph("4. Resultados na minha base", styles["Heading2"]),
            Paragraph(
                f"Acuracia: {custom_result['accuracy']:.2%}<br/>"
                f"Precisao: {custom_result['precision']:.2%}<br/>"
                f"Recall: {custom_result['recall']:.2%}<br/>"
                f"F1-score: {custom_result['f1']:.2%}",
                styles["BodyText"],
            ),
            Paragraph(
                "Matriz de confusao:<br/>"
                f"{custom_result['matrix'].tolist()}",
                styles["BodyText"],
            ),
        ])

    content.extend([
        Paragraph("5. Analise e discussao", styles["Heading2"]),
        Paragraph(
            "Compare as metricas da base de teste e da base propria. "
            "Analise os erros e procure padroes de iluminacao, distancia, "
            "angulo, fundo, camera, racas ou qualidade da imagem.",
            styles["BodyText"],
        ),
        Paragraph(
            "Verifique as previsoes muito confiantes e avalie o fenomeno "
            "de mudanca de distribuicao (domain shift) entre os dados de treino "
            "e as imagens reais.",
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


def show_dataset_preview(root: Path) -> None:
    st.subheader("Imagens da base")
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
        columns = st.columns(min(5, len(images)))
        for column, image_path in zip(columns, images[:5]):
            column.image(str(image_path), caption=image_path.name, use_container_width=True)


st.title("🐱 CNN — Gato ou Cachorro")
st.caption(
    "Aplicação do experimento do material: data augmentation, treino, validação, "
    "teste e avaliação da sua própria base."
)

with st.sidebar:
    st.header("Configuração")
    dataset_zip = st.file_uploader(
        "Dataset principal (.zip)", type="zip", key="dataset"
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

if dataset_zip is None and st.session_state.dataset_dir is None:
    if DEFAULT_DATASET_ZIP.is_file():
        folder = Path(tempfile.mkdtemp(prefix="cnn_dataset_"))
        try:
            st.session_state.dataset_dir = str(
                safe_extract(DEFAULT_DATASET_ZIP, folder, dataset_zip=True)
            )
            st.session_state.dataset_name = DEFAULT_DATASET_ZIP.name
            st.info(
                "Usando minha_base.zip: 20 fotos divididas automaticamente "
                "em treino, validação e teste."
            )
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
            st.session_state.dataset_dir = str(
                safe_extract(dataset_zip, folder, dataset_zip=True)
            )
            st.session_state.dataset_name = dataset_zip.name
            st.success("Dataset principal carregado.")
        except (ValueError, zipfile.BadZipFile) as error:
            shutil.rmtree(folder, ignore_errors=True)
            st.session_state.dataset_dir = None

            if DEFAULT_DATASET_ZIP.is_file():
                folder = Path(tempfile.mkdtemp(prefix="cnn_dataset_"))
                try:
                    st.session_state.dataset_dir = str(
                        safe_extract(
                            DEFAULT_DATASET_ZIP, folder, dataset_zip=True
                        )
                    )
                    st.session_state.dataset_name = DEFAULT_DATASET_ZIP.name
                    st.warning(
                        "Este ZIP não é válido para o treino. "
                        "minha_base.zip foi carregada automaticamente."
                    )
                except (ValueError, zipfile.BadZipFile):
                    shutil.rmtree(folder, ignore_errors=True)
                    st.error(str(error))
            else:
                st.error(str(error))

if st.session_state.dataset_dir is None:
    st.info("Envie o ZIP do dataset para começar. A estrutura esperada está no README.")
    st.stop()

dataset_dir = Path(st.session_state.dataset_dir)
st.write(
    f"**Dataset:** `{st.session_state.get('dataset_name', 'dataset.zip')}` · "
    f"classes: `{', '.join(CLASS_NAMES)}` · imagem: `64 × 64 RGB`"
)
show_dataset_preview(dataset_dir)

if train_button:
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
                "epocas": grid_epochs,
                "learning_rate": grid_lr,
                "acc_validacao": float(grid_history.history["val_accuracy"][-1]),
                "perda_validacao": float(grid_history.history["val_loss"][-1]),
                "acc_teste": result["accuracy"],
            }
        )
        current_key = (rows[-1]["acc_validacao"], -rows[-1]["perda_validacao"])
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_model = model
            best_history = grid_history
            best_result = result
        progress.progress(index / len(configurations), text=f"{index}/9 concluídas")
    st.session_state.grid = pd.DataFrame(rows)
    st.session_state.model = best_model
    st.session_state.history = best_history
    st.session_state.test_result = best_result
    st.session_state.selected_config = st.session_state.grid.sort_values(
        ["acc_validacao", "perda_validacao"], ascending=[False, True]
    ).iloc[0].to_dict()

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

st.divider()
st.header("Relatorio")
if "test_result" in st.session_state:
    pdf_report = generate_report_pdf()
    st.download_button(
        "Gerar e baixar relatorio PDF",
        data=pdf_report,
        file_name="relatorio_experimento_cnn.pdf",
        mime="application/pdf",
        type="primary",
    )
else:
    st.info("Treine uma configuracao antes de gerar o relatorio.")

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
    st.write("Envie um ZIP contendo as pastas `cachorro` e `gato`.")
    custom_zip = st.file_uploader("Minha base (.zip)", type="zip", key="custom")
    if custom_zip is not None and "model" in st.session_state:
        custom_root = Path(tempfile.mkdtemp(prefix="cnn_custom_"))
        try:
            custom_dir = safe_extract(custom_zip, custom_root)
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
        except (ValueError, zipfile.BadZipFile) as error:
            st.error(str(error))
        finally:
            shutil.rmtree(custom_root, ignore_errors=True)
