
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """
    Converts image directories into training and testing DataLoaders.

    Handles the full dataset creation process including directory mapping, 
    preprocessing via transforms, and batching. Simplifies the data pipeline 
    setup for image classification models.

    Args:
        train_dir (str): Directory for training samples.
        test_dir (str): Directory for testing samples.
        transform (transforms.Compose): Preprocessing steps to apply.
        batch_size (int): Samples per batch.
        num_workers (int): Workers for parallel loading (default is CPU count).

    Returns:
        tuple: (train_dataloader, test_dataloader, class_names)
  """
  # Use ImageFolder to create dataset
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False, # don't need to shuffle test data
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names
