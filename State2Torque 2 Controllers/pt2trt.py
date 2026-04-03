import torch
import tensorrt as trt
from TCN_Header_Model import TCNModel  # Update to your actual model class

# def pt_to_trt(pt_model_path, trt_engine_path, hyperparam_config, fp16_mode=False):
#     # Load PyTorch model
#     model = TCNModel(hyperparam_config).eval()
#     state_dict = torch.load(pt_model_path, map_location="cpu", weights_only=False)
#     model.load_state_dict(state_dict)
#     model.cuda()

#     # Export to ONNX
#     onnx_path = trt_engine_path.replace('.trt', '.onnx')
#     dummy_input = torch.randn(
#         1,  # batch size
#         hyperparam_config['input_size'],
#         hyperparam_config['window_size']
#     ).cuda()

#     torch.onnx.export(
#         model,
#         dummy_input,
#         onnx_path,
#         input_names=['input'],
#         output_names=['output'],
#         opset_version=11,
#     )
#     print(f"[INFO] ONNX model saved to {onnx_path}")

#     # TensorRT engine building
#     logger = trt.Logger(trt.Logger.WARNING)
#     builder = trt.Builder(logger)
#     network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
#     parser = trt.OnnxParser(network, logger)

#     # Parse ONNX model
#     with open(onnx_path, 'rb') as model_file:
#         if not parser.parse(model_file.read()):
#             print("[ERROR] Failed to parse ONNX model:")
#             for i in range(parser.num_errors):
#                 print(f"  {parser.get_error(i)}")
#             raise RuntimeError("ONNX model parsing failed.")

#     # Build TensorRT engine
#     config = builder.create_builder_config()
#     config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

#     if fp16_mode:
#         if builder.platform_has_fast_fp16:
#             config.set_flag(trt.BuilderFlag.FP16)
#             print("[INFO] Building engine in FP16 mode.")
#         else:
#             print("[WARNING] FP16 not supported on this platform. Building in FP32 mode.")

#     print("[INFO] Building TensorRT engine...")
#     serialized_engine = builder.build_serialized_network(network, config)

#     if serialized_engine is None:
#         raise RuntimeError("TensorRT engine build failed.")

#     # Save engine to file
#     with open(trt_engine_path, "wb") as f:
#         f.write(serialized_engine)

#     print(f"[SUCCESS] TensorRT engine saved at: {trt_engine_path}")


# # Example usage
# # pt_model = '/home/metamobility2/Changseob/biotorque_controller/trained_model/baseline_TCN-wo_AB05_Maria/baseline_TCN-wo_AB05_Maria.pt'
# # trt_engine = '/home/metamobility2/Changseob/biotorque_controller/trained_model/baseline_TCN-wo_AB05_Maria/baseline_TCN-wo_AB05_Maria.trt'

# # pt_model = '/home/metamobility2/Jimin/baseline_TCN_14subjects-wo_AB01_Jimin/baseline_TCN_14subjects-wo_AB01_Jimin.pt'
# # trt_engine = '/home/metamobility2/Jimin/baseline_TCN_14subjects-wo_AB01_Jimin/baseline_TCN_14subjects-wo_AB01_Jimin.trt'

# # pt_model = '/home/metamobility2/Jimin/TrainedModels/SI_AB02_LGRARD/SI_AB02_LGRARD.pt'
# # trt_engine = '/home/metamobility2/Jimin/TrainedModels/SI_AB02_LGRARD/SI_AB02_LGRARD.trt'

# pt_model = '/home/metamobility2/Jimin/TrainedModels/SI_AB01_Jimin_ResumeTest/SI_AB01_Jimin_ResumeTest_epoch_5.pt'
# trt_engine = '/home/metamobility2/Jimin/TrainedModels/SI_AB01_Jimin_ResumeTest/SI_AB01_Jimin_ResumeTest_epoch_5.trt'


# hyperparam_config = {
#     'wandb_project_name': 'baseline_TCN',
#     'wandb_session_name': 'baseline_TCN_nature_hyperparam',
#     'input_size': 8, # 12 for IMU (right, left), 2 for hip angle and velocity
#     'output_size': 1, # 1 for right hip torque
#     'architecture': 'TCN',
    
#     'transfer_learning': False,
#     'dataset_proportion': 1.0, # dataset proportion for training
    
#     'epochs': 30,
#     'batch_size': 32,
#     'init_lr': 5e-5,
#     'dropout': 0.15,
#     'validation_split': 0.1,
#     'window_size': 95,
#     'number_of_layers': 2,
#     'num_channels': [80, 80, 80, 80, 80],
#     'kernel_size': 5,
#     'dilations': [1, 2, 4, 8, 16],
#     'number_of_workers': 10,
# }



try:
    from torchsummary import summary
    TORCHSUMMARY_AVAILABLE = True
except ImportError:
    print("[WARNING] torchsummary not available. Model summary will be skipped.")
    TORCHSUMMARY_AVAILABLE = False

def pt_to_trt(pt_model_path, trt_engine_path, hyperparam_config, fp16_mode=False):
    # Load PyTorch model
    model = TCNModel(hyperparam_config).eval()
    state_dict = torch.load(pt_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.cuda()

    # Print model summary if torchsummary is available
    if TORCHSUMMARY_AVAILABLE:
        print("\n" + "="*60)
        print("MODEL SUMMARY")
        print("="*60)
        try:
            summary(model, input_size=(hyperparam_config['input_size'], hyperparam_config['window_size']))
        except Exception as e:
            print(f"[WARNING] Could not generate model summary: {e}")
        print("="*60 + "\n")
    else:
        # Manual parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[INFO] Total parameters: {total_params:,}")
        print(f"[INFO] Trainable parameters: {trainable_params:,}")
        print(f"[INFO] Model size (MB): {total_params * 4 / 1024 / 1024:.2f}\n")

    # Export to ONNX
    onnx_path = trt_engine_path.replace('.trt', '.onnx')
    dummy_input = torch.randn(
        1,  # batch size
        hyperparam_config['input_size'],
        hyperparam_config['window_size']
    ).cuda()

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
    )
    print(f"[INFO] ONNX model saved to {onnx_path}")

    # TensorRT engine building
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX model
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print("[ERROR] Failed to parse ONNX model:")
            for i in range(parser.num_errors):
                print(f"  {parser.get_error(i)}")
            raise RuntimeError("ONNX model parsing failed.")

    # Build TensorRT engine
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if fp16_mode:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] Building engine in FP16 mode.")
        else:
            print("[WARNING] FP16 not supported on this platform. Building in FP32 mode.")

    print("[INFO] Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed.")

    # Save engine to file
    with open(trt_engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"[SUCCESS] TensorRT engine saved at: {trt_engine_path}")


# Example usage
# pt_model = '/home/metamobility2/Changseob/biotorque_controller/trained_model/baseline_TCN-wo_AB05_Maria/baseline_TCN-wo_AB05_Maria.pt'
# trt_engine = '/home/metamobility2/Changseob/biotorque_controller/trained_model/baseline_TCN-wo_AB05_Maria/baseline_TCN-wo_AB05_Maria.trt'

# pt_model = '/home/metamobility2/Jimin/TrainedModels/SI_AB01_Jimin_ResumeTest/SI_AB01_Jimin_ResumeTest.pt'
# trt_engine = '/home/metamobility2/Jimin/TrainedModels/SI_AB01_Jimin_ResumeTest/SI_AB01_Jimin_ResumeTest.trt'

# pt_model = '/home/metamobility2/Jimin/TrainedModels/SI_AB01_Jimin_LGRARD_EntireDataset/SI_AB01_Jimin_LGRARD_EntireDataset.pt'
# trt_engine = '/home/metamobility2/Jimin/TrainedModels/SI_AB01_Jimin_LGRARD_EntireDataset/SI_AB01_Jimin_LGRARD_EntireDataset.trt'

# pt_model = '/home/metamobility2/Jimin/TrainedModels/SI_AB02_Rajiv_LGRARD_EntireDataset/SI_AB02_Rajiv_LGRARD_EntireDataset.pt'
# trt_engine = '/home/metamobility2/Jimin/TrainedModels/SI_AB02_Rajiv_LGRARD_EntireDataset/SI_AB02_Rajiv_LGRARD_EntireDataset.pt'

pt_model = '/home/metamobility2/Jimin/TrainedModels/SI_AB02_Rajiv_Selective/SI_AB02_Rajiv_Selective.pt'
trt_engine = '/home/metamobility2/Jimin/TrainedModels/SI_AB02_Rajiv_Selective/SI_AB02_Rajiv_Selective.trt'

# Base hyperparameters
hyperparam_config = {
    'wandb_project_name': 'Biotorque_LGRARD',
    'wandb_session_name': 'SI_AB02_Rajiv_Selective',
    'input_size': 8, # 12 for IMU (right, left), 2 for hip angle and velocity
    'output_size': 1, # 1 for right hip torque
    'architecture': 'TCN',
    
    'transfer_learning': False,
    'dataset_proportion': 1.0, # dataset proportion for training
    
    'epochs': 30,
    'batch_size': 32,
    'init_lr': 5e-6,
    'dropout': 0.15,
    'validation_split': 0.1,
    'window_size': 95,
    'number_of_layers': 2,
    'num_channels': [80, 80, 80, 80, 80],
    'kernel_size': 5,
    'dilations': [1, 2, 4, 8, 16],
    'number_of_workers': 10,
}


pt_to_trt(pt_model, trt_engine, hyperparam_config, fp16_mode=False)
