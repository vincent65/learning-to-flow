"""
Inference pipeline for FCLF.

Single-image attribute manipulation.
"""

import torch
import clip
from PIL import Image
import numpy as np
from typing import Optional, Dict
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vector_field import VectorFieldNetwork
from models.clip_decoder import CLIPDecoder


class FCLFInference:
    """FCLF inference pipeline for attribute manipulation."""

    ATTRIBUTES = ['Smiling', 'Young', 'Male', 'Eyeglasses', 'Mustache']

    def __init__(
        self,
        vector_field_checkpoint: str,
        decoder_checkpoint: str,
        device: str = None
    ):
        """
        Initialize inference pipeline.

        Args:
            vector_field_checkpoint: Path to trained vector field checkpoint
            decoder_checkpoint: Path to trained decoder checkpoint
            device: Device to use
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load CLIP
        print("Loading CLIP...")
        self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device=self.device)
        self.clip_model.eval()

        # Load vector field
        print("Loading vector field...")
        vf_checkpoint = torch.load(vector_field_checkpoint, map_location=self.device)
        vf_config = vf_checkpoint['config']['model']

        self.vector_field = VectorFieldNetwork(
            embedding_dim=vf_config['embedding_dim'],
            num_attributes=vf_config['num_attributes'],
            hidden_dim=vf_config['hidden_dim']
        )
        self.vector_field.load_state_dict(vf_checkpoint['model_state_dict'])
        self.vector_field = self.vector_field.to(self.device)
        self.vector_field.eval()

        # Load decoder
        print("Loading decoder...")
        decoder_checkpoint_data = torch.load(decoder_checkpoint, map_location=self.device)
        decoder_config = decoder_checkpoint_data['config']['model']

        self.decoder = CLIPDecoder(
            embedding_dim=decoder_config['embedding_dim'],
            img_size=decoder_config['img_size']
        )
        self.decoder.load_state_dict(decoder_checkpoint_data['model_state_dict'])
        self.decoder = self.decoder.to(self.device)
        self.decoder.eval()

        print("Inference pipeline ready!")

    def encode_image(self, image_path: str) -> torch.Tensor:
        """
        Encode image to CLIP embedding.

        Args:
            image_path: Path to image

        Returns:
            embedding: [512] CLIP embedding
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.clip_model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.squeeze(0)

    def decode_embedding(self, embedding: torch.Tensor) -> Image.Image:
        """
        Decode CLIP embedding to image.

        Args:
            embedding: [512] CLIP embedding

        Returns:
            PIL Image
        """
        with torch.no_grad():
            # Add batch dimension if needed
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)

            # Decode
            img_tensor = self.decoder(embedding)

            # Convert to PIL image
            img_tensor = (img_tensor + 1) / 2  # [-1, 1] -> [0, 1]
            img_tensor = img_tensor.clamp(0, 1)
            img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            return Image.fromarray(img_np)

    def flow_embedding(
        self,
        embedding: torch.Tensor,
        target_attributes: Dict[str, int],
        num_steps: int = 10,
        step_size: float = 0.1,
        return_trajectory: bool = False
    ):
        """
        Flow embedding toward target attributes.

        Args:
            embedding: [512] CLIP embedding
            target_attributes: Dict mapping attribute name to value (0 or 1)
            num_steps: Number of flow steps
            step_size: Step size
            return_trajectory: Whether to return full trajectory

        Returns:
            flowed_embedding or (flowed_embedding, trajectory)
        """
        # Create attribute vector
        attr_vector = torch.zeros(len(self.ATTRIBUTES))
        for attr_name, value in target_attributes.items():
            if attr_name in self.ATTRIBUTES:
                idx = self.ATTRIBUTES.index(attr_name)
                attr_vector[idx] = value

        attr_vector = attr_vector.unsqueeze(0).to(self.device)
        embedding = embedding.unsqueeze(0) if embedding.dim() == 1 else embedding

        with torch.no_grad():
            if return_trajectory:
                trajectory = self.vector_field.get_trajectory(
                    embedding,
                    attr_vector,
                    num_steps=num_steps,
                    step_size=step_size
                )
                flowed_embedding = trajectory[:, -1, :]
                return flowed_embedding.squeeze(0), trajectory.squeeze(0)
            else:
                flowed_embedding = self.vector_field.flow(
                    embedding,
                    attr_vector,
                    num_steps=num_steps,
                    step_size=step_size
                )
                return flowed_embedding.squeeze(0)

    def transform_image(
        self,
        image_path: str,
        target_attributes: Dict[str, int],
        num_steps: int = 10,
        step_size: float = 0.1,
        return_intermediate: bool = False
    ):
        """
        Complete pipeline: image -> flow -> image.

        Args:
            image_path: Path to input image
            target_attributes: Target attributes
            num_steps: Number of flow steps
            step_size: Step size
            return_intermediate: Return intermediate images along trajectory

        Returns:
            transformed_image or (transformed_image, intermediate_images)
        """
        # Encode
        embedding = self.encode_image(image_path)

        # Flow
        if return_intermediate:
            _, trajectory = self.flow_embedding(
                embedding,
                target_attributes,
                num_steps=num_steps,
                step_size=step_size,
                return_trajectory=True
            )

            # Decode all intermediate embeddings
            images = []
            for i in range(trajectory.shape[0]):
                img = self.decode_embedding(trajectory[i])
                images.append(img)

            return images[-1], images
        else:
            flowed_embedding = self.flow_embedding(
                embedding,
                target_attributes,
                num_steps=num_steps,
                step_size=step_size
            )

            # Decode
            transformed_image = self.decode_embedding(flowed_embedding)

            return transformed_image


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="FCLF Inference")
    parser.add_argument("--image", type=str, required=True,
                        help="Input image path")
    parser.add_argument("--vector_field", type=str, required=True,
                        help="Vector field checkpoint")
    parser.add_argument("--decoder", type=str, required=True,
                        help="Decoder checkpoint")
    parser.add_argument("--output", type=str, default="output.png",
                        help="Output image path")
    parser.add_argument("--attributes", type=str, required=True,
                        help="Target attributes (e.g., 'Smiling:1,Young:0')")
    parser.add_argument("--num_steps", type=int, default=10,
                        help="Number of flow steps")
    parser.add_argument("--step_size", type=float, default=0.1,
                        help="Flow step size")
    parser.add_argument("--device", type=str, default=None,
                        help="Device")

    args = parser.parse_args()

    # Parse attributes
    target_attrs = {}
    for pair in args.attributes.split(','):
        name, value = pair.split(':')
        target_attrs[name.strip()] = int(value.strip())

    # Initialize pipeline
    pipeline = FCLFInference(
        vector_field_checkpoint=args.vector_field,
        decoder_checkpoint=args.decoder,
        device=args.device
    )

    # Transform image
    print(f"Transforming {args.image}...")
    print(f"Target attributes: {target_attrs}")

    transformed_image = pipeline.transform_image(
        image_path=args.image,
        target_attributes=target_attrs,
        num_steps=args.num_steps,
        step_size=args.step_size
    )

    # Save
    transformed_image.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
