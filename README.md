# TP2- Detetor de Objetos
Sistemas Avançados de Visualização Industrial (SAVI) - Grupo 4 - Universidade de Aveiro - 2023/24


<h3 align="center">Wait it's all Garlic? </h3>

<div align="center">
  <img width="500" alt="Captura de ecrã 2024-02-06, às 16 46 47" src="https://github.com/Ze6000/Trabalho_SAVI_2/assets/148572902/2fd3798c-c4d6-4f92-9f76-e05f619008ee">
</div> 

## Índice

- [Introdução](#introduction)
- [Bibliotecas Usadas](#libraries-used)
- [Execução](#installation)
- [Autores](#authors)

---
## Introdução

<p align="justify"> No âmbito da Unidade Curricular de SAVI, foi criado um programa capaz de detetar objetos que se aproximem da câmera, reconhecendo em diferentes cenários. <br> </p>





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
    
- **Intertools**
  - Descrição: A Itertools é um módulo em python, usado para iterar estruturas de dados que podem ser superadas usando um loop for.
  - Instalação:
    ```bash
    pip install more-itertools
    ```
---
## Execução

O programa pode ser executado seguindo os seguintes passos:

1. Clonar o repositório:
```bash
git clone https://github.com/ze6000/Trabalho_SAVI_2
```
2. Fazer o download do dataset

https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset/

3. No ficheiro Split_dataset é necessário alterar a linha de código 18 com o path onde foram instaladas as imagens.
   
4. Preparar DataSet:
```bash
cd /Trabalho_SAVI_2/Split_dataset
./main.py
```  
5. Iniciar Treino: 
```bash
cd /Trabalho_SAVI_2/Training
./main.py
```
Para aplicar métricas de performance, assim quer o treino terminar, executar 
```bash
./test.py
```


Se os passos acima foram seguidos, o programa deve correr sem problemas.

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
