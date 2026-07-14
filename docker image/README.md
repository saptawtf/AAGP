# AAGP Predictor (Anti-Aging Peptide Predictor)

This tool is fully containerized using **Docker**.

## Prerequisites

You don't need to install Python, complex machine learning libraries, or resolve dependency conflicts. The only requirement is Docker.

* **Install Docker Desktop:**
    * **Windows / Mac:** Download and install from the [official Docker website](https://www.docker.com/products/docker-desktop/).
* Make sure the Docker application is running in the background before proceeding.

---

## Quick Start

The easiest way to use the AAGP Predictor is to pull the pre-built Docker image directly from Docker Hub.

### 1. Prepare your input data
Create a folder on your computer and place your input `.fasta` file inside it. 

### 2. Run the prediction
Open your terminal (Command Prompt, PowerShell, or bash) and run the following command. 

```bash
docker run --rm -v "YOUR_FOLDER:/app/data/output" takoyakiyee/aagp-predictor-01:latest python main_predict.py --dataset DS1 --input /app/data/output/YOUR_FASTA_NAME.fasta
```
If you have more than two FASTA files:
```bash
docker run --rm -v "YOUR_FOLDER:/app/data/output" takoyakiyee/aagp-predictor-01:latest python main_predict.py --dataset DS1 --input /app/data/output/YOUR_FIRST_FASTA_NAME.fasta /app/data/output/YOUR_SECOND_FASTA_NAME.fasta
```
### 2.1 Note:
1. Replace `<span style="color:red">YOUR_FOLDER</span>` with the actual absolute path to your local folder. 
2. Replace `span style="color:red">YOUR_FASTA_NAME.fasta</span>` with the exact name of your FASTA file.
3. Replace `<span style="color:red">YOUR_FIRST_FASTA_NAME.fasta</span>` and `<span style="color:red">YOUR_SECOND_FASTA_NAME.fasta</span>` with the exact name of your FASTA file.
3. When `<span style="color:red">--dataset DS1</span>`: the program will use models trained on DS1, corresponding features and their normalization scaler to process data and perform prediction.
4. When `<span style="color:red">-dataset DS2</span>`: the program will use models trained on DS2, corresponding features and their normalization scaler to process data and perform prediction.



### 📂 Output Files
binary_vector.csv -- The prediction output in binary format (1 for positive and 0 for negative).

probability.csv -- The prediction probability estimate.

### Docker Hub
Website: https://hub.docker.com/r/takoyakiyee/aagp-predictor-01
