# TP2- Detetor de Objetos
Sistemas Avançados de Visualização Industrial (SAVI) - Grupo 4 - Universidade de Aveiro - 2023/24

## Índice

- [Introdução](#introduction)
- [Bibliotecas Usadas](#libraries-used)
- [Instalação](#installation)
- [Explicação do Código](#code-explanation)
- [Autores](#authors)

---
## Introdução

<p align="justify"> No âmbito da Unidade Curricular de SAVI, foi criado um programa capaz de detetar objetos que se aproximem da câmera, reconhecendo em diferentes cenários. <br> </p>

[Video.webm](https://github.com/joaonogueiro/TP1_SAVI/assets/114345550/8f64f7c6-c3a3-4698-b44e-39805258fb1)

<p align="center">
Vídeo ilustrativo do funcionamento do programa 
</p>


---
## Bibliotecas Usadas

Para a criação deste programa, recorreu-se à utilização de algumas bibliotecas. Estas serão brevemente explicadas abaixo.

- **Open3D**
  - Descrição: Open3D é uma biblioteca de código aberto, com a interface de software sofisticada para a implementação de dados 3D. 
  - Instalação:
    ```bash
    pip install open3d 
    ```

- **Opens CV**
  - Descrição: OpenCV é uma biblioteca que existe em Python desenhada para resolver problemas de _computer vision_. 
  - Instalação:
    ```bash
    sudo apt-get install python3-opencv
    ```

- **Tqdme**
  - Descrição: Esta é uma biblioteca de Python para adicionar barra de progresso..
  - Instalação:
    ```bash
    pip install tqdm
    ```

- **Pytorch**
  - Descrição: Esta bibioteca permite introduzir uma sequência de intruções num programa que podem ser executadas independentemente do restante processo.
  - Instalação: https://pytorch.org/get-started/locally/
   
  
- **Scikit-learn**
  - Descrição: Esta bibioteca também conhecida como sklearn permite implementar modelos de aprendizado de máquina e modelagem estatística.
  - Instalação:
    ```bash
    pip install -U scikit-learn
    ```

---
## Instalação

O programa pode ser instalado seguindo os seguintes passos:

1. Clonar o repositório:
```bash
git clone https://github.com/ze6000/Trabalho_SAVI_2
```
2. Alterar a diretória do projeto:
```bash
cd Trabalho_SAVI_2
```
3. Correr o programa:
```bash
./main.py
```

Se os passos acima foram seguidos, o programa deve correr sem problemas.


---
## Explicação do Código 

<p align="justify"> O código começa por verificar se existe alguma informação na base de dados, e se assim se verificar, lê a mesma. De seguida, inicializa a câmera e começa a tentar encontrar deteções de caras com os comandos abaixo, baseados na biblioteca <b>Face-Recognition</b>.</p>

```python
# Find all the faces and face encodings in the current frame of video
face_locations = face_recognition.face_locations(rgb_small_frame)
face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
```
<p align="justify">É na base destes comandos, que se baseiam em modelos extreamemte eficientes treinados com deep learning, que o programa irá funcionar. Aqui são encontradas todas as localizações de caras no _frame_ e sofrem um _encoding_ para posteriormente serem comparadas com as caras guardadas na <i>Database</i>. Se o programa encontrar um nível de parecença elevado com alguma das informações da base de dados, irá reconhecer e cumprimentar a pessoa detetada. Além disto, o programa ainda faz o seguimento de cada pessoa.</p>
<p align="justify">Ao mesmo tempo, um menu no Terminal estará a correr em pararelo (usando a biblioteca <b>Threading</b>) onde se poderá dar nome às pessoas detetadas como desconhecidas e ainda alterar o nome de qualquer pessoa presente na <i>Database</i>.</p>

---
## Autores

Estes foram os contribuidores para este projeto:

- **[Afonso Miranda](https://github.com/afonsosmiranda)**
  - Informação:
    - Email: afonsoduartem@ua.pt
    - Número Mecanográfico: 100090

- **[Carolina Francisco ](https://github.com/Carolf27)**
  - Informação:
    - Email: carolinaf@ua.pt
    - Número Mecanográfico: 98303

- **[José Duarte](https://github.com/Ze6000)**
  - Informação:
    - Email: josemduarte@ua.pt
    - Número Mecanográfico: 103892
