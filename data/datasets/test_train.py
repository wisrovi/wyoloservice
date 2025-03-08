from train_yolo import train_run


if __name__ == "__main__":
    
    
    
    
    
    
    resultado = train_run(
        config_path="/datasets/test/colorball.v8i.multiclass/config_train.yaml",
        trial_number=1,
        verbose=False,
        # script_path="lib/train_yolo/",
    )
    if resultado is not None:
        print(f"Resultado del entrenamiento: {resultado}")