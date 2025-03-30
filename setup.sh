#!/bin/bash

# Create required directories
mkdir -p data/templates/packages
mkdir -p data/templates/sheets
mkdir -p data/roi_mappings/packages
mkdir -p data/roi_mappings/sheets
mkdir -p models/classifier
mkdir -p models/ocr

# Download pre-trained OCR models if they don't exist
if [ ! -f "models/ocr/det_model.pdmodel" ]; then
    echo "Downloading OCR detection model..."
    wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar -O det_model.tar
    tar -xf det_model.tar -C models/ocr
    mv models/ocr/ch_PP-OCRv4_det_infer/inference.pdmodel models/ocr/det_model.pdmodel
    mv models/ocr/ch_PP-OCRv4_det_infer/inference.pdiparams models/ocr/det_model.pdiparams
    rm -rf models/ocr/ch_PP-OCRv4_det_infer
    rm det_model.tar
fi

if [ ! -f "models/ocr/rec_model.pdmodel" ]; then
    echo "Downloading OCR recognition model..."
    wget https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar -O rec_model.tar
    tar -xf rec_model.tar -C models/ocr
    mv models/ocr/en_PP-OCRv4_rec_infer/inference.pdmodel models/ocr/rec_model.pdmodel
    mv models/ocr/en_PP-OCRv4_rec_infer/inference.pdiparams models/ocr/rec_model.pdiparams
    rm -rf models/ocr/en_PP-OCRv4_rec_infer
    rm rec_model.tar
fi

# Create default configuration file
if [ ! -f "config/system_config.json" ]; then
    echo "Creating default configuration..."
    cat > config/system_config.json << EOF
{
    "hardware": {
        "cpu_cores": 6,
        "gpu_memory_fraction": 0.8,
        "use_gpu": true,
        "use_arm_optimization": true
    },
    "paths": {
        "templates_dir": "data/templates",
        "roi_mappings_dir": "data/roi_mappings",
        "models_dir": "models"
    },
    "performance": {
        "max_processing_time_ms": 500,
        "target_classification_time_ms": 50,
        "target_template_matching_time_ms": 100,
        "target_ocr_time_ms": 300
    },
    "ocr": {
        "use_det": true,
        "use_cls": false,
        "use_rec": true,
        "det_model_path": "models/ocr/det_model",
        "rec_model_path": "models/ocr/rec_model",
        "rec_batch_num": 6,
        "det_db_thresh": 0.3,
        "det_db_box_thresh": 0.5,
        "rec_char_dict_path": "en_dict.txt"
    }
}
EOF
fi

# Build Docker image
docker-compose build

echo "Setup complete! You can now run the system with 'docker-compose up'"
