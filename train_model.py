from model import create_model
from data_preprocessing import load_data, batch_generator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Parameters
batch_size = 32
epochs = 10

def train_model():
    data = load_data()
    
    train_data, val_data = train_test_split(data, test_size=0.2)
    
    print(f'Training samples: {len(train_data)}, Validation samples: {len(val_data)}')
    
    model = create_model()
    
    history = model.fit(
        batch_generator(train_data, batch_size),
        steps_per_epoch=len(train_data) // batch_size,
        epochs=epochs,
        validation_data=batch_generator(val_data, batch_size, training=False),
        validation_steps=len(val_data) // batch_size
    )

    model.save('model.h5')
    
    plt.figure(figsize=(10, 8))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()
    
    print('Training Complete!')

if __name__ == '__main__':
    train_model()