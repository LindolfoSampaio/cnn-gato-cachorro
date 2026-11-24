# Aplicação CNN: gato ou cachorro

Aplicação Streamlit e notebook didático para o experimento de classificação de
gatos e cachorros, com imagens de `64 x 64`, data augmentation e saída binária.

## Executar a aplicação

Requer Python 3.12 ou superior:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

O arquivo de treino `dataset_gato_cachorro.zip` não está no GitHub porque
ultrapassa o limite de tamanho por arquivo. Obtenha o ZIP da disciplina e
coloque-o ao lado de `app.py` para carregamento automático, ou envie-o pela
barra lateral do aplicativo. A estrutura deve conter `training_set/` e
`test_set/`, ambos com pastas `cachorro/` e `gato/`. O app valida 2.000 imagens
por classe no treino e 500 por classe no teste, separando 20% do treino para
validação.

O arquivo `minha_base.zip` incluído no projeto é usado apenas na avaliação
final, nunca no treinamento. Deve conter 10 fotos próprias de gatos e 10 de
cachorros, organizadas em pastas `gato/` e `cachorro/`, conforme as instruções
do experimento.

## Notebook e resultados

Para executar o notebook localmente, coloque o ZIP de treino ao lado dele;
no Colab, envie o arquivo para `/content` antes de executar as células. A grade
das nove configurações está habilitada por padrão. O CSV, as curvas e o modelo
selecionado pela validação ficam no projeto. A aplicação carrega esses
resultados automaticamente quando o ZIP padrão está disponível.

O app também permite testar imagens individuais, usar a câmera, avaliar um ZIP
com as duas classes e baixar o relatório PDF, o CSV e o modelo selecionado.
