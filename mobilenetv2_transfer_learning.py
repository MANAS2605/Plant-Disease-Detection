import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint

def build_transfer_learning_model(num_classes=38):
    """
    Builds a MobileNetV2 transfer learning model for crop disease classification.
    Assuming the PlantVillage dataset has 38 classes by default.
    """
    
    # ==========================================
    # 1. Model Setup
    # ==========================================
    # Set input shape
    input_shape = (224, 224, 3)
    
    # Load the MobileNetV2 architecture pre-trained on ImageNet weights.
    # Set include_top=False to remove the original classification layer.
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Set base_model.trainable = False to freeze the convolutional base and preserve learned features.
    base_model.trainable = False
    
    # ==========================================
    # 2. Custom Classifier Head
    # ==========================================
    inputs = Input(shape=input_shape)
    
    # MobileNetV2 expects pixel values to be in [-1, 1], so we can use its preprocess_input
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Pass inputs through the base model.
    # training=False prevents BatchNormalization layers from updating their statistics during feature extraction.
    x = base_model(x, training=False)
    
    # Add a GlobalAveragePooling2D layer to reduce the spatial dimensions.
    x = GlobalAveragePooling2D()(x)
    
    # Add a Dropout layer (rate of 0.2) to prevent overfitting on the new dataset.
    x = Dropout(0.2)(x)
    
    # Add a final Dense layer with Softmax activation (units = num_classes)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # ==========================================
    # 3. Training & Compilation
    # ==========================================
    # Use the Adam optimizer with a low learning rate (e.g., 0.0001) to ensure stable convergence.
    optimizer = Adam(learning_rate=0.0001)
    
    # Use CategoricalCrossentropy for the loss function.
    loss_fn = CategoricalCrossentropy()
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    return model, base_model

def fine_tune_model(model, base_model):
    """
    4. Fine-Tuning (Optional Step)
    Unfreeze the top 20 layers of the base model and re-train with a very small learning rate 
    to fine-tune the model to the specific textures of plant leaves.
    """
    print("\n--- Starting Fine-Tuning Setup ---")
    
    # Unfreeze the base model
    base_model.trainable = True
    
    # Total layers in MobileNetV2 base is ~154
    # Calculate the layer index from which we want to start fine-tuning
    layers_to_unfreeze = 20
    freeze_until = len(base_model.layers) - layers_to_unfreeze
    
    # Freeze all layers before the 'freeze_until' index
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
        
    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True
        
    print(f"Total layers in base model: {len(base_model.layers)}")
    print(f"Frozen layers: {freeze_until}")
    print(f"Unfrozen layers (top layers): {layers_to_unfreeze}")

    # Re-compile the model with a very small learning rate to avoid destroying learned weights
    model.compile(
        optimizer=Adam(learning_rate=1e-5), # 1e-5 is a very small learning rate
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Define number of classes in your dataset (PlantVillage usually has 38)
    NUM_CLASSES = 38
    
    print("Building Feature Extraction Model...")
    model, base_model = build_transfer_learning_model(num_classes=NUM_CLASSES)
    model.summary()
    
    # Implement ModelCheckpoint to save the best performing model based on validation accuracy
    checkpoint_callback = ModelCheckpoint(
        filepath='best_mobilenetv2_crop_disease.keras', # modern keras extension
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # --- PROPOSED TRAINING LOOP (Feature Extraction) ---
    print("\n[NOTE]: You would run model.fit() here for Feature Extraction.")
    '''
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=10, 
        callbacks=[checkpoint_callback]
    )
    '''
    
    # Prepare model for fine-tuning
    model = fine_tune_model(model, base_model)
    model.summary()
    
    # --- PROPOSED TRAINING LOOP (Fine-Tuning) ---
    print("\n[NOTE]: You would run model.fit() here for Fine-Tuning.")
    '''
    fine_tune_checkpoint = ModelCheckpoint(
        filepath='best_mobilenetv2_crop_disease_finetuned.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    total_epochs = 10 + 5 # initial_epochs + fine_tune_epochs
    history_fine = model.fit(
        train_dataset,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=validation_dataset,
        callbacks=[fine_tune_checkpoint]
    )
    '''
    print("Script fully defined.")
