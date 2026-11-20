# Aplicação CNN: gato ou cachorro

Aplicação web com uma CNN para classificar gatos e cachorros. Usa imagens de
`64 x 64`, data augmentation e saída binária, considerando `gato` como classe
positiva.

## Executar

Requer Python 3.12:

```powershell
& "C:\Users\netin\AppData\Local\Programs\Python\Python312\python.exe" -m pip install -r requirements.txt
& "C:\Users\netin\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run app.py
```

Por padrão, a aplicação usa `minha_base.zip`, com 10 fotos de cada classe. Ela
divide automaticamente as 20 imagens em:

```text
12 imagens de treino
4 imagens de validação
4 imagens de teste
```

A aplicação também permite:

- enviar uma ou várias imagens JPG, JPEG ou PNG;
- tirar uma foto pela câmera;
- enviar um ZIP com as pastas `cachorro` e `gato`.
- gerar e baixar um relatório em PDF.
