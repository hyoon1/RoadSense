<!-- https://github.com/othneildrew/Best-README-Template/ -->
<a name="readme-top"></a>



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Lakshay1505/RoadSense">
    <img src="assets/logo.png">
  </a>

  <h3 align="center">RoadSense</h3>

  <p align="center">
    Real-time Detection of Road Damage
    <br />
    <a href="https://github.com/Lakshay1505/RoadSense"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <!--<a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a> -->
  </p>
</div>




<!-- ABOUT THE PROJECT -->
## About The Project

Potholes and poor road conditions pose significant risks to road safety, causing accidents, vehicle damage, and traffic delays. According to the report by CAA (Canadian Automobile Association), poor-quality roads cost Canadian drivers approximately `$3 billion` annually, with an average of `$126` per vehicle per year in additional operating costs. This translates to over `$1,250` in extra costs over the 10-year lifespan of a vehicle.

The goal of this project is to develop and implement a <b>real-time road damage detection and severity assessment system</b> using computer vision technology accessible through smartphones, dash cams, and traffic cameras. Additionally, the system will predict maintenance needs based on historical data to optimize road repair schedules.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With
<!-- https://ileriayo.github.io/markdown-badges/ -->

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Azure](https://img.shields.io/badge/azure-%230072C6.svg?style=for-the-badge&logo=microsoftazure&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

```py
git clone https://github.com/hyoon1/RoadSense.git
cd RoadSense
```

### Prerequisites

- Python 3.9
- tensorflow 2.10
- pytorch 2.3
- wandb

### Installation
Install the required packages
```py
pip install -r requirements.txt
```
Install the YOLOv8 package from Ultralytics
```py
pip install ultralytics
```
<!--
1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->



<!-- USAGE EXAMPLES -->
## Usage

To use this project, follow the steps below:

1. Prepare the dataset by filtering and converting it to the YOLO format.
2. Train the YOLOv8 model using the prepared dataset.
3. Run inference using the trained model.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ML usecase 1: Road Damage Detection -->
## ML usecase1: Road Damage Detection
### Dataset
The model is trained on the RDD2022 dataset. You can download the dataset from the official RDD2022 repository.
### Dataset Links

- [GitHub Repository for Road Damage Detection Dataset](https://github.com/sekilab/RoadDamageDetector)
- [Japan Dataset](https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Japan.zip)
- [Filtered Japan Dataset (D00, D10, D20, D40 labels only)](https://stuconestogacon-my.sharepoint.com/:u:/g/personal/hyoon6442_conestogac_on_ca/ETxbhuMBQX5OhqSsITEUUYgBcrU2wipogzRVdDbYcjgI5Q?e=SfISCj)
### Data Preparation
1. Filter the dataset to include only specific labels(D00, D10, D20, D40) using the provided 'filter_dataset.py' script.
   ```py
   python filter_dataset.py
   ```
2. Convert the filtered RDD2022 dataset to the YOLO format using the provided 'yolo_data_converter.py' script. This script will also split the dataset into training and validation sets.
   ```py
   python yolo_data_converter.py
   ```
### Model Training and Inference
To train the YOLOv8 model on the RDD2022 dataset, use the 'train.py' script.
For inference, use the 'predict.py' script with the best saved model.

You can download the pretrained models from the links below:
- **YOLOv8 Model (.pt)**: [Download YOLOv8 .pt model](https://stuconestogacon-my.sharepoint.com/:u:/g/personal/hyoon6442_conestogac_on_ca/ESVuF62OmXBHs6g-WHxlEAQBwbEY52ymRiN1iYyHKEVk6g?e=NkmJzx)  
- **TFLite Model (.tflite)**: [Download TFLite .tflite model](https://stuconestogacon-my.sharepoint.com/:u:/g/personal/hyoon6442_conestogac_on_ca/Eb5bXqKWsLpClH_zlrHCUVYByzGubulB1bLantX3HchC3w?e=2Qjv94)

1. Training the model: The results will be saved in the './runs/detect/train/' directory.
   ```py
   python train.py
   ```
2. Running Inference:
   ```py
   python predict.py
   ```

### Docker
The Dockerfile is located in the root of the folder. The main purpose of Docker is to run the Streamlit app in a container. The Python file for the Streamlit app is located at `./streamlit/app.py`. The Docker configuration files are located in the `./docker/` directory, including the Python requirements file (requirements.txt).

1. Bulding a Docker image:
   ```
   docker build -t roadsense:tag_id .
   ```
2. Check Dockder images:
   ```
   docker images
   ```
3. Running Docker container:
   ```
   docker run -p 8501:8501 roadsense:tag_id

Dockerhub repository name: [king138786/roadsense](https://hub.docker.com/r/king138786/roadsense)


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE 
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->



<!-- Authorizers -->
## Authorizers

- [Lakshay1505](https://github.com/Lakshay1505)
- [imdarshik](https://github.com/imdarshik)
- [guggg](https://github.com/guggg)
- Hosang Yoon
- Yeji Kim



## Project Link: 
[RoadSense](https://github.com/Lakshay1505/RoadSense)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!