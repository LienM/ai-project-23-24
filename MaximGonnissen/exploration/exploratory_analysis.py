import multiprocessing as mp
import pathlib
import time
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

from utils.utils import load_data, get_data_path, DataFileNames


def describe_data(df: pd.DataFrame, name: str, output_path_: pathlib.Path,
                  thread_start_time: Optional[float] = None) -> None:
    """
    Describe a dataframe and write the results to a markdown file.
    :param df: Dataframe to describe
    :param name: Name of the dataframe
    :param output_path_: Path to write the markdown file to
    :param thread_start_time: Time when the thread started, if applicable
    :return:
    """
    time_start = time.time()
    output_file = output_path_ / f'{name}_description.md'

    print(
        f'[ ] Describing data from {name}...{f" (thread started in {time.time() - thread_start_time:.2f} seconds)" if thread_start_time is not None else ""}')

    with open(output_file, 'w') as f:
        f.write(f'# {name}\n\n')
        f.write(f'## Shape\n\n')
        f.write(f'Rows: {df.shape[0]}\n\n')
        f.write(f'Columns: {df.shape[1]}\n\n')
        f.write(f'## Head\n\n')
        f.write(f'{df.head().to_markdown()}\n\n')
        f.write(f'## Tail\n\n')
        f.write(f'{df.tail().to_markdown()}\n\n')
        f.write(f'## Describe\n\n')
        f.write(f'{df.describe().to_markdown()}\n\n')
        f.write(f'## Missing values\n\n')
        f.write(f'{df.isna().sum().to_markdown()}\n\n')
        f.write(f'## Duplicated values sum\n\n')
        f.write(f'{df.duplicated().sum()}\n\n')
        f.write(f'## Total values\n\n')
        f.write(f'{df.count().to_markdown()}\n\n')
        f.write(f'## Unique values\n\n')
        f.write(f'{df.nunique().to_markdown()}\n\n')
        f.write(f'## Unique percentage\n\n')
        f.write(f'{(df.nunique() / df.shape[0] * 100).to_markdown()}\n\n')
        f.write(f'## Column types\n\n')
        f.write(f'{df.dtypes.to_markdown()}\n\n')

    nan_output_file = output_path_ / f'{name}_nan_rows.md'
    with open(nan_output_file, 'w') as f:
        f.write(f'# {name}\n\n')
        f.write(f'## Shape\n\n')
        f.write(f'Rows: {df[df.isna().any(axis=1)].shape[0]}\n\n')
        f.write(f'## NaN rows\n\n')
        f.write(f'{df[df.isna().any(axis=1)].to_markdown()}\n\n')

    print(f'[X] Described data from {name} in {time.time() - time_start:.2f} seconds.'
          f' ({time.time() - thread_start_time:.2f} seconds since thread start)' if thread_start_time is not None else '')


def add_image_info(image_path: pathlib.Path) -> dict:
    """
    Create a dictionary with the image name, width and height.
    :param image_path: Path to the image
    :return: Dictionary with the image name, width and height
    """
    with Image.open(image_path) as image:
        width, height = image.size
        image_name = image_path.name
        return {'image': image_name, 'width': width, 'height': height}


def describe_images(path: pathlib.Path, output_path: pathlib.Path, mp_pool_count: int = 1) -> None:
    """
    Describe all images found recursively in the given path and write the results to a markdown file.
    :param path: Path to the images
    :param output_path: Path to write the markdown file to
    :param mp_pool_count: Number of processes to use for multiprocessing
    :return:
    """
    time_start = time.time()
    if output_path.suffix != '.md':
        output_path = output_path / 'images_description.md'

    print(f'[ ] Describing images...')

    subfolders = [subfolder for subfolder in path.glob('*') if subfolder.is_dir()]
    images = [image for subfolder in subfolders for image in subfolder.glob('*.jpg') if image.is_file()]
    images += [image for image in path.glob('*.jpg') if image.is_file()]

    print(f'\tFound {len(images)} images in {len(subfolders)} subfolders.')

    df = pd.DataFrame(columns=['image', 'width', 'height'])
    df = df.astype({'image': 'object', 'width': 'int64', 'height': 'int64'})

    with mp.Pool(processes=mp_pool_count) as pool:
        df = df._append(pool.map(add_image_info, images), ignore_index=True)

    with open(output_path, 'w') as f:
        f.write(f'# Images\n\n')
        f.write(f'## Shape\n\n')
        f.write(f'Rows: {df.shape[0]}\n\n')
        f.write(f'Columns: {df.shape[1]}\n\n')
        f.write(f'## Head\n\n')
        f.write(f'{df.head().to_markdown()}\n\n')
        f.write(f'## Tail\n\n')
        f.write(f'{df.tail().to_markdown()}\n\n')
        f.write(f'## Describe width\n\n')
        f.write(f'{df.width.describe().to_markdown()}\n\n')
        f.write(f'## Describe height\n\n')
        f.write(f'{df.height.describe().to_markdown()}\n\n')
        median_width = df.width.median()
        median_height = df.height.median()
        f.write(f'## Median width\n\n')
        f.write(f'{median_width} ({df[df.width == median_width].shape[0] / df.shape[0] * 100:.2f}%)\n\n')
        f.write(f'## Median height\n\n')
        f.write(f'{median_height} ({df[df.height == median_height].shape[0] / df.shape[0] * 100:.2f}%)\n\n')
        f.write(f'## Median width and height\n\n')
        f.write(
            f'{df[(df.width == median_width) & (df.height == median_height)].shape[0] / df.shape[0] * 100:.2f}%\n\n')

    print(f'[X] Described images in {time.time() - time_start:.2f} seconds.')


def create_plot(df: pd.DataFrame, name: str, output_path: pathlib.Path, binwidth: float = 2, height: int = 5,
                aspect: int = 2, plot_width: int = 15, plot_height: int = 10, rotation: float = 60, dpi: int = 400,
                **kwargs) -> None:
    """
    Create a plot for the given column in the given dataframe and save it to the given path.
    :param df: Dataframe to create the plot for
    :param name: Name of the column to create the plot for
    :param output_path: Path to save the plot to
    :param binwidth: Width of the bins
    :param height: Height of the plot
    :param aspect: Aspect ratio of the plot
    :param plot_width: Width of the plot
    :param plot_height: Height of the plot
    :param rotation: Rotation of the x-axis labels
    :param dpi: DPI of the plot
    :param kwargs: Additional arguments to pass to the plot
    """
    print(f'[ ] Creating plot for {name}...')
    df = df.copy()

    sns.displot(data=df, x=name, binwidth=binwidth, height=height, aspect=aspect, **kwargs)
    plt.gcf().set_size_inches(plot_width, plot_height)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(output_path / f'{name}.png', dpi=dpi, bbox_inches='tight', **kwargs)
    plt.close()
    print(f'[X] Created plot for {name}.')


if __name__ == '__main__':
    script_start_time = time.time()

    # Calculate max number of processes to allow for multiprocessing
    _mp_pool_count = max(mp.cpu_count() - 1, 1)
    print(f'Using {_mp_pool_count} cores for multiprocessing.')

    # Initialize Seaborn theme
    sns.set_theme()

    data_path = get_data_path()

    # Find data
    h_and_m_path = data_path / DataFileNames.HNM_DIR
    articles_path = h_and_m_path / DataFileNames.ARTICLES
    customers_path = h_and_m_path / DataFileNames.CUSTOMERS
    sample_submission_path = h_and_m_path / DataFileNames.SAMPLE_SUBMISSION
    transactions_train_path = h_and_m_path / DataFileNames.TRANSACTIONS_TRAIN
    images_path = h_and_m_path / DataFileNames.IMAGES_DIR

    # Create output and plot directories
    output_path = data_path / DataFileNames.OUTPUT_DIR
    if not output_path.exists():
        output_path.mkdir()

    plot_path = output_path / DataFileNames.PLOTS_DIR
    if not plot_path.exists():
        plot_path.mkdir()

    articles = load_data(articles_path)
    customers = load_data(customers_path)
    sample_submission = load_data(sample_submission_path)
    transactions_train = load_data(transactions_train_path)

    # Plot total sales over time
    sales_channel_id = transactions_train.groupby(['t_dat', 'sales_channel_id']).size().reset_index(name='count')
    sales_channel_id['t_dat'] = pd.to_datetime(sales_channel_id['t_dat'])
    sales_channel_id = sales_channel_id.pivot(index='t_dat', columns='sales_channel_id', values='count')
    sales_channel_id = sales_channel_id.resample('M').sum()
    fig, ax = plt.subplots(figsize=(25, 10))
    sales_channel_id.plot(kind='bar', stacked=True, ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Total sales')
    ax.set_title('Total sales over time, grouped by channel')
    ax.legend(['Online', 'Store'])
    plt.tight_layout()
    plt.savefig(plot_path / 'sales_channel_id.png', dpi=400, bbox_inches='tight')

    # Plot relevant data as Seaborn plots
    with mp.Pool(processes=_mp_pool_count) as _pool:
        _pool.starmap(create_plot, [(customers, 'age', plot_path),
                                    (articles, 'product_type_name', plot_path, 0, 1, 5, 2, 25, 10, 75),
                                    (articles, 'product_group_name', plot_path),
                                    (articles, 'graphical_appearance_name', plot_path),
                                    (articles, 'colour_group_name', plot_path, 0, 2, 5, 2, 15, 10, 90),
                                    (articles, 'perceived_colour_value_name', plot_path),
                                    (articles, 'perceived_colour_master_name', plot_path),
                                    (articles, 'index_name', plot_path), (articles, 'index_group_name', plot_path),
                                    (articles, 'section_name', plot_path, 0, 1, 5, 2, 20, 10, 75),
                                    (articles, 'garment_group_name', plot_path),
                                    (articles, 'department_name', plot_path, 0, 1, 5, 2, 35, 10, 90),
                                    (customers, 'club_member_status', plot_path),
                                    (customers, 'fashion_news_frequency', plot_path)])

    # Describe data
    thread_start_time = time.time()
    with mp.Pool(processes=_mp_pool_count) as _pool:
        _pool.starmap(describe_data, [(articles, 'articles', output_path, thread_start_time),
                                      (customers, 'customers', output_path, thread_start_time),
                                      (sample_submission, 'sample_submission', output_path, thread_start_time),
                                      (transactions_train, 'transactions_train', output_path, thread_start_time)])

    # Describe images
    describe_images(images_path, output_path, _mp_pool_count)

    print(f'[X] Finished script in {time.time() - script_start_time:.2f} seconds.')
