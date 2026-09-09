"""Convert PyTorch checkpoint to TorchScript format for C++ deployment."""

import torch
import sys
from pathlib import Path


def convert_checkpoint_to_torchscript(checkpoint_path: str, output_path: str):
    """Convert RSL-RL checkpoint to TorchScript format.
    
    This extracts the actual policy network from the checkpoint and converts it.
    
    Args:
        checkpoint_path: Path to checkpoint .pt file
        output_path: Path to output .pt file (TorchScript format)
    """
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Input checkpoint: {checkpoint_path}")
    print(f"Output path: {output_path}")
    
    # Load checkpoint
    print("\n[1/3] Loading checkpoint...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Loaded successfully. Type: {type(checkpoint)}")
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return False
    
    # Extract and reconstruct policy network
    print("\n[2/3] Extracting and reconstructing policy network...")
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print(f"✅ Found 'model_state_dict' in checkpoint")
            
            # Extract actor state_dict (remove 'actor.' prefix)
            actor_state = {k.replace('actor.', ''): v 
                          for k, v in model_state.items() if k.startswith('actor.')}
            
            # Infer dimensions from weights
            # actor.0.weight shape is [512, num_obs]
            actor_first_weight_shape = actor_state['0.weight'].shape
            num_obs = actor_first_weight_shape[1]
            
            # actor.6.weight shape is [num_actions, 128]
            actor_last_weight_shape = actor_state['6.weight'].shape
            num_actions = actor_last_weight_shape[0]
            
            print(f"   Detected dimensions:")
            print(f"     - num_obs: {num_obs}")
            print(f"     - num_actions: {num_actions}")
            print(f"     - hidden_dims: (512, 256, 128)")
            print(f"     - activation: elu")
            
            # Create a simple MLP network that matches the actor architecture
            # (actor is a simple sequential MLP, not the full ActorCritic)
            print(f"\n   Creating actor network...")
            import torch.nn as nn
            
            # Build the network to match the actor architecture
            layers = []
            activation_fn = nn.ELU
            
            # Input layer
            layers.append(nn.Linear(num_obs, 512))
            layers.append(activation_fn())
            
            # Hidden layers
            layers.append(nn.Linear(512, 256))
            layers.append(activation_fn())
            
            layers.append(nn.Linear(256, 128))
            layers.append(activation_fn())
            
            # Output layer (no activation)
            layers.append(nn.Linear(128, num_actions))
            
            actor = nn.Sequential(*layers)
            
            # Load the state dict
            actor.load_state_dict(actor_state)
            print(f"   ✅ Loaded actor weights successfully")
            
            # Set to eval mode
            actor.eval()
            
        else:
            print("❌ Checkpoint format not recognized")
            return False
        
    except Exception as e:
        print(f"❌ Failed to extract/reconstruct policy: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Convert to TorchScript
    print("\n[3/3] Converting to TorchScript...")
    try:
        # Create dummy input for tracing
        dummy_input = torch.randn(1, num_obs)
        
        print(f"   Tracing actor network with dummy input shape: {dummy_input.shape}")
        scripted_model = torch.jit.trace(actor, dummy_input)
        print(f"✅ Successfully converted to TorchScript")
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the scripted model
        torch.jit.save(scripted_model, output_path)
        print(f"✅ Saved TorchScript model to: {output_path}")
        
        # Verify the file was created and show file size
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size / 1024 / 1024
            print(f"   File size: {file_size:.2f} MB")
            print("\n✅ Conversion completed successfully!")
            return True
        else:
            print(f"❌ Output file was not created")
            return False
        
    except Exception as e:
        print(f"❌ Failed to convert to TorchScript: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    input_checkpoint = "logs/rsl_rl/go2_velocity/2026-01-25_00-10-32/model_9999.pt"
    output_model = "logs/rsl_rl/go2_velocity/2026-01-25_00-10-32/model_9999_torchscript.pt"
    
    if len(sys.argv) > 1:
        input_checkpoint = sys.argv[1]
    if len(sys.argv) > 2:
        output_model = sys.argv[2]
    
    success = convert_checkpoint_to_torchscript(input_checkpoint, output_model)
    sys.exit(0 if success else 1)
