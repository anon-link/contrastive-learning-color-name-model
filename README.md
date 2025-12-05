# Contrastive Learning Color Name Model

This is the repository for the paper "Contrastive Learning for Large-scale Color-Name Dataset: Tackling Sparsity with Negative Sampling"

## Project Overview

This project implements a contrastive learning-based color name model that learns the mapping between RGB color values and color names. The model adopts a dual-tower architecture, combining pre-trained Transformers and NCF (Neural Collaborative Filtering) architecture, supporting bidirectional recommendations from color to name and name to color.

## Key Features

- **Dual-Tower Architecture**: RGB encoder and name encoder separately process color and text information
- **Pre-trained Transformer**: Uses BERT and other pre-trained models for semantic understanding of color names
- **NCF Architecture**: Integrates neural collaborative filtering mechanism to improve matching accuracy
- **Negative Sampling**: Probability sampling strategy based on LAB distance to enhance model generalization
- **Multi-Loss Functions**: Supports combination of contrastive learning loss, binary classification loss, RGB generation loss, and other loss functions

## Requirements

- Python >= 3.8
- CUDA >= 11.0 (optional, for GPU acceleration)

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. Download Pre-trained Models (Optional)

Pre-trained Transformer models (such as BERT) will be automatically downloaded on first run, or you can manually download them to the `models-pretrained/` directory.

## Data Preparation

1. Prepare a CSV format data file containing the following columns:
   - `r`: Red value (0-255)
   - `g`: Green value (0-255)
   - `b`: Blue value (0-255)
   - `term`: Color name (string)

2. Name the data file as `responses_cleaned_all.csv` and place it in the project root directory

## Usage

### Training the Model

```bash
python color-name-model.py train
```

**Running in Background**: For long-running training sessions, you can run the training process in the background:

**Linux/Mac**:
```bash
nohup python color-name-model.py train &
```

**Windows PowerShell**:
```powershell
Start-Process python -ArgumentList "color-name-model.py", "train" -NoNewWindow
# Or run directly in a new window:
Start-Process python -ArgumentList "color-name-model.py", "train"
```

**Windows CMD**:
```cmd
start python color-name-model.py train
# Or run in background (minimized):
start /min python color-name-model.py train
```

**Monitoring GPU Usage**: To monitor GPU utilization during training:

**Linux/Mac**:
```bash
watch -n 1 nvidia-smi
```

**Windows PowerShell**:
```powershell
# Run nvidia-smi in a loop
while ($true) { Clear-Host; nvidia-smi; Start-Sleep -Seconds 1 }
```

**Windows CMD**:
```cmd
# Run nvidia-smi in a loop (press Ctrl+C to stop)
:loop
cls
nvidia-smi
timeout /t 1 /nobreak >nul
goto loop
```

The training process will:
- Automatically load and preprocess data
- Train RGB encoder, name encoder, and NCF model
- Save the best model to `models-pretrained/` directory
- Generate loss curve plots

### Evaluating the Model

```bash
python color-name-model.py evaluate
```

The evaluation process will calculate:
- **Color-to-Name Accuracy**: Top-1, Top-3, Top-5, Top-10 accuracy
- **Name-to-Color Accuracy**: Recommendation accuracy based on CIELAB distance
- **Text-to-Color Accuracy**: Generation accuracy of RGB generator

## Configuration Parameters

You can modify the following configuration parameters in `color-name-model.py`:

```python
# Architecture Configuration
use_ncf_architecture = True  # Whether to use NCF architecture
use_16d_feature = True  # Whether to use 16-dimensional features (RGB+Lab+HCL+HSL+CMYK)
use_freezed_weight = True  # Whether to freeze pre-trained model weights
use_negtive_sample = True  # Whether to use negative samples

# Training Configuration
used_learning_rate = 1e-4  # Learning rate
used_num_epochs = 30  # Number of training epochs
batch_size = 1024  # Batch size
```

## Model Architecture

The model consists of the following main components:

1. **RGBEncoder**: RGB color encoder that encodes RGB values into 64-dimensional embedding vectors
2. **NameEncoder**: Name encoder that encodes color names into 64-dimensional embedding vectors using pre-trained Transformers
3. **NCFModel**: Neural Collaborative Filtering model that fuses RGB and name embeddings to output matching scores
4. **RGBGenerator**: RGB generator that generates RGB color values from name embeddings

## Output Files

After training, the following files will be generated:

- `models-pretrained/model_best_*.pt`: Best model weights
- `models-pretrained/plot_loss_curve_*.png`: Loss curve plots
- `train_*.log`: Training log files
- `models/preprocessed_data.pkl`: Preprocessed data

## Notes

1. The first run requires downloading pre-trained models, which may take a long time
2. Training requires significant GPU memory, recommend using a GPU with at least 16GB VRAM
3. The data file should contain sufficient number of samples (recommend at least 100,000 samples)
4. You can adjust batch size and number of training epochs according to actual conditions

### Web Application

The project includes a web interface for interactive color-name translation. The web application provides a user-friendly interface for:

- **Color-to-Name Prediction**: Select a color (via color picker or RGB input) and get top-10 recommended color names with confidence scores
- **Name-to-Color Generation**: Enter a color name and generate corresponding RGB color values
- **Color Recommendations**: Get multiple color recommendations based on a color name

#### Starting the Web Application

1. **Navigate to the website directory**:
```bash
cd website
```

2. **Activate the virtual environment** (if not already activated):
```bash
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Start the web server**:

**Using GPU (default)**:
```bash
python web_app.py --device gpu --port 5000
```

**Using CPU**:
```bash
python web_app.py --device cpu --port 5000
```

**Running in Background (Linux/Mac)**:
```bash
nohup python web_app.py --device cpu --port 5000 &
```

**Running in Background (Windows PowerShell)**:
```powershell
Start-Process python -ArgumentList "web_app.py", "--device", "cpu", "--port", "5000" -NoNewWindow
```

4. **Access the web interface**: Open your browser and navigate to `http://localhost:5000`

#### Web Application Features

- **Color Picker**: Interactive color picker for visual color selection
- **RGB Input**: Manual RGB value input (supports formats: `255,215,123` or `255 215 123`)
- **Real-time Prediction**: Instant color name predictions with confidence scores
- **Color Generation**: Generate RGB colors from color names
- **Top-K Recommendations**: Get top-10 recommendations for both color-to-name and name-to-color tasks

#### Web Application Requirements

- The web application requires a trained model. Make sure you have:
  - A trained model file in `website/models/` directory (e.g., `model_best_*.pt`)
  - Pre-trained BERT model in `website/models-pretrained/bert-base-uncased/` directory
  - Preprocessed data file `website/models/preprocessed_data.pkl` (optional, will be generated if missing)

- The first startup may take some time to:
  - Load the model weights
  - Precompute embeddings for faster inference
  - Initialize the pre-trained transformer model

#### API Endpoints

The web application provides the following REST API endpoints:

- `GET /`: Main web interface
- `POST /predict_color_name`: Predict color names from RGB values
  - Request body: `{"r": 255, "g": 0, "b": 0}`
  - Response: `{"success": true, "results": [{"name": "red", "score": 0.95}, ...]}`
- `POST /generate_quick_color`: Generate RGB color from color name
  - Request body: `{"color_name": "red"}`
  - Response: `{"success": true, "result": {"rgb": [255, 0, 0], "hex": "#ff0000", ...}}`
- `POST /recommend_colors`: Get color recommendations from color name
  - Request body: `{"color_name": "red"}`
  - Response: `{"success": true, "results": [{"rgb": [255, 0, 0], "hex": "#ff0000", ...}, ...]}`

## Citation

If you use this code, please cite the related paper:

```bibtex
@article{xx,
  title={Contrastive Learning for Large-scale Color-Name Dataset: Tackling Sparsity with Negative Sampling},
}
```

## License

Please see the LICENSE file for details.
