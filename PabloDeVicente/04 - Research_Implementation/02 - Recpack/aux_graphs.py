import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap



# Seaborn setup
sns.set_theme(style="whitegrid")
sns.set_style('whitegrid') #same?

inferno = 'inferno'
blueish = 'YlGnBu'
palette = blueish

palette_colormap = plt.cm.YlGnBu
num_colors = 20  
colors = palette_colormap(np.linspace(0, 1, num_colors))
reversed_colors = colors[::-1]
reversed_palette = ListedColormap(reversed_colors)


color_codes = {
    "Black": "#222222",
    "Blue": "#ADD8E6",
    "White": "#E6E6E6",
    "Beige": "#FEFCC9",
    "Grey": "#808080",
    "Pink": "#FFC0CB",
    "Lilac Purple": "#C29DC2",
    "Red": "#FF6E66",
    "Mole": "#B86D29",
    "Orange": "#FF9955",
    "Metal": "#B5B5B5",
    "Brown": "#876E58",
    "Turquoise": "#E4FFF1",
    "Yellow": "#FFFAA3",
    "Khaki green": "#8FBC8F",
    "Green": "#90EE90",
    "Yellowish Green": "#9ACD32",
    "Bluish Green": "#00FA9A"
}

def sns_bar(grouped_data):
    plt.figure(figsize=(5, 3))  
    sns.barplot(x=grouped_data.index, y=grouped_data.values,palette=palette,hue=grouped_data.index,legend=False)      
    # plt.xticks(rotation=45) 
    return plt


def sns_brightness_count(recommended_items):
    plt.figure(figsize=(5,3))
    colour_value_name = ["Bright", "Light", "Dusty Light", "Medium", "Medium Dusty", "Dark"]
    plot = sns.countplot(y='perceived_colour_value_name', data=recommended_items, palette=palette, order=colour_value_name)
    return plot

def plt_brightness_pie(recommended_items):

    perceived_colors = recommended_items['perceived_colour_value_name'].value_counts(normalize=True)
    colors_threshold = 0.01  # Set the threshold to exclude segments with less than 1%

    # Filter colors that have less than 1% occurrence
    filtered_colors = perceived_colors[perceived_colors >= colors_threshold]
    labels_to_display = filtered_colors.index
    sizes_to_display = filtered_colors.values

    plt.figure(figsize=(5, 3))  # Maintain a larger figure size
    
    # colors = sns.color_palette(palette, len(filtered_colors))[::-1] 
    colors = sns.color_palette("pastel", len(filtered_colors)) 
    
    plt.pie(sizes_to_display, labels=labels_to_display, colors=colors, startangle=140)
    # plt.pie(sizes_to_display, labels=labels_to_display, colors=colors, startangle=140, autopct='%1.1f%%')
    
    plt.title('Perceived Colors Distribution', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()

    return plt

def sns_color_ord(recommended_items):

    known_colors = recommended_items[~recommended_items['perceived_colour_master_name'].isin(['undefined', 'Unknown'])]

    color_counts = known_colors['perceived_colour_master_name'].value_counts()
    ordered_colors = color_counts.index.tolist() 

    plt.figure(figsize=(5,3))
    sns.countplot(y='perceived_colour_master_name', data=known_colors, palette=color_codes, order=ordered_colors)
   
    return plt


def sns_color_grp(recommended_items):
    # Filter out undefined or unknown colors
    known_colors = recommended_items[~recommended_items['perceived_colour_master_name'].isin(['undefined', 'Unknown'])]

    # Grouping similar colors based on provided color codes
    similar_colors = {
        "Black": ["Black", "Grey"],
        "White": ["White", "Beige"],
        "Blue": ["Blue", "Turquoise"],
        "Pink": ["Pink", "Lilac Purple"],
        "Red": ["Red", "Mole"],
        "Orange": ["Orange"],
        "Metal": ["Metal", "Brown"],
        "Yellow": ["Yellow", "Yellowish Green"],
        "Green": ["Green", "Khaki green", "Bluish Green"]
    }

    # Create a reverse lookup dictionary for colors to group
    reverse_lookup = {}
    for key, values in similar_colors.items():
        for value in values:
            reverse_lookup[value] = key

    # Map colors to their grouped categories
    known_colors['grouped_color'] = known_colors['perceived_colour_master_name'].map(reverse_lookup)

    # Countplot for grouped colors
    plt.figure(figsize=(5, 3))
    sns.countplot(y='grouped_color', data=known_colors, palette=color_codes)
    plt.title('Grouped Colors')
    plt.xlabel('Count')
    plt.ylabel('Color Group')

    return plt

def sns_season_color(recommended_items):
    
    winter_colors = {
        "Black": "#222222",
        "Blue": "#ADD8E6",
        "Grey": "#808080",
        "Lilac Purple": "#C29DC2",
        "Mole": "#B86D29",
        "Turquoise": "#E4FFF1"
    }
    autumn_colors = {
        "Beige": "#FEFCC9",
        "Brown": "#876E58",
        "Orange": "#FF9955",
        "Metal": "#B5B5B5",
        "Red": "#FF6E66"
    }
    summer_colors = {
        "White": "#E6E6E6",
        "Pink": "#FFC0CB",
        "Yellow": "#FFFAA3",
        "Khaki green": "#8FBC8F",
        "Green": "#90EE90",
        "Yellowish Green": "#9ACD32",
        "Bluish Green": "#00FA9A"
    }

    seasonal_data = pd.DataFrame({
        'Color': list(winter_colors.keys()) + list(autumn_colors.keys()) + list(summer_colors.keys()),
        'Hex': list(winter_colors.values()) + list(autumn_colors.values()) + list(summer_colors.values()),
        'Season': (['Winter'] * len(winter_colors)) + (['Autumn'] * len(autumn_colors)) + (['Summer'] * len(summer_colors))
    })

    # Group data by color and season, count occurrences, and merge it with the seasonal_data
    color_counts = recommended_items['perceived_colour_master_name'].value_counts().reset_index()
    color_counts.columns = ['Color', 'Count']
    color_counts = color_counts.merge(seasonal_data, on='Color', how='left')

    # Aggregate counts by season
    seasonal_counts = color_counts.groupby('Season')['Count'].sum().reset_index()

    plt.figure(figsize=(5, 3))
    sns.barplot(x='Season', y='Count', data=seasonal_counts, palette=['#E2b153', '#882928', '#87CEEB'])
    plt.title('Aggregate Counts of Seasonal Colors')
    plt.xlabel('Season')
    plt.ylabel('Aggregate Count')
    plt.tight_layout()

    return plt


def sns_fabric_usd(recommended_items):

    columns = [
        'jeans', 'cotton', 'wool', 'polyester', 'silk',
        'denim', 'linen', 'spandex', 'rayon', 'nylon',
        'leather', 'suede'
    ]

    counts = recommended_items[columns].sum().sort_values(ascending=False)
    counts_df = counts.reset_index()
    counts_df.columns = ['Fabric', 'Count']

    plt.figure(figsize=(20, 3))

    sns.barplot(x='Fabric', y='Count', data=counts_df, palette='pastel')
    plt.title('Aggregate of fabric types')
    plt.xlabel('Fabric')
    plt.ylabel('Aggregate Count')
    plt.show()
        
    return plt



def number_of_repeats(recommended_items):

    plt.figure(figsize=(5, 3))
    sns.histplot(data=recommended_items, x='number_of_repeats', bins=range(0, 21),palette = palette)
    plt.xlabel('Number of times recommended')
    plt.ylabel('Frequency')
    plt.title('Number of times an item has been recommended')
    plt.grid(True)
    plt.show()
    return plt

def number_of_repeats_pop(recommended_items):

    plt.figure(figsize=(5, 3))
    plt.hist(recommended_items['number_of_repeats'], bins=range(0, 45000), edgecolor='black')  # 20 bins from 0 to 20
    plt.xlabel('Number of Repeats')
    plt.ylabel('Frequency')
    plt.title('Distribution of Number of Repeats')
    plt.grid(True)
    plt.show()
    return plt



def sns_fabric_usd(recommended_items, articles_df, normalized=1):
    columns = [
        'jeans', 'cotton', 'wool', 'polyester', 'silk',
        'denim', 'linen', 'spandex', 'rayon', 'nylon',
        'leather', 'suede'
    ]

    plt.figure(figsize=(20, 5))

    if normalized:
        # Calculate fabric totals and sort the fabrics by their totals in recommended_items
        fabric_totals = articles_df[columns].sum()
        fabric_totals_sorted = fabric_totals

        # Create a palette with the sorted fabric order for consistent colors
        palette = sns.color_palette("pastel", n_colors=len(fabric_totals_sorted))

        for fabric, color in zip(fabric_totals_sorted.index, palette):
            # Calculate the total count for the current fabric type from the articles DataFrame
            fabric_total = articles_df[fabric].sum()

            # Calculate normalized values for the current fabric type by dividing by its total count
            normalized_counts = recommended_items[fabric].sum() / fabric_total

            # Create a DataFrame to hold the normalized counts for the current fabric type
            normalized_counts_df = pd.DataFrame({'Fabric': [fabric], 'Normalized_Count': [normalized_counts]})

            # Create a bar plot for the current fabric type with a specific color
            sns.barplot(x='Fabric', y='Normalized_Count', data=normalized_counts_df, color=color)

        plt.title('Aggregate of fabric types (Normalized)')
        plt.ylabel('Normalized Aggregate Count')
    else:
        # Sum the counts of each fabric type
        counts = recommended_items[columns].sum()
        counts_df = counts.reset_index()
        counts_df.columns = ['Fabric', 'Count']

        # Create a bar plot for the fabric types without normalization
        sns.barplot(x='Fabric', y='Count', data=counts_df, palette='pastel')

        plt.title('Aggregate of fabric types')
        plt.ylabel('Aggregate Count')

    plt.xlabel('Fabric')
    plt.xticks(rotation=45)
    plt.show()

    return plt

#-------------------old stuff

#aux functions for color mapping
def color_mapping(recommended_items):

    color_mapping = {
    'Black': 'Black',
    'White': 'White',
    'Off White': 'White',
    'Light Beige': 'Beige',
    'Beige': 'Beige',
    'Grey': 'Grey',
    'Light Blue': 'Blue',
    'Light Grey': 'Grey',
    'Dark Blue': 'Blue',
    'Dark Grey': 'Grey',
    'Pink': 'Pink',
    'Dark Red': 'Red',
    'Greyish Beige': 'Beige',
    'Light Orange': 'Orange',
    'Silver': 'Silver',
    'Gold': 'Gold',
    'Dark Pink': 'Pink',
    'Yellowish Brown': 'Brown',
    'Blue': 'Blue',
    'Light Pink': 'Pink',
    'Light Turquoise': 'Turquoise',
    'Yellow': 'Yellow',
    'Greenish Khaki': 'Green',
    'Dark Yellow': 'Yellow',
    'Other Pink': 'Pink',
    'Dark Purple': 'Purple',
    'Red': 'Red',
    'Transparent': 'Other',
    'Dark Green': 'Green',
    'Other Red': 'Red',
    'Turquoise': 'Turquoise',
    'Dark Orange': 'Orange',
    'Other': 'Other',
    'Orange': 'Orange',
    'Dark Beige': 'Beige',
    'Light Green': 'Green',
    'Other Orange': 'Orange',
    'Purple': 'Purple',
    'Light Red': 'Red',
    'Light Yellow': 'Yellow',
    'Green': 'Green',
    'Light Purple': 'Purple',
    'Dark Turquoise': 'Turquoise',
    'Other Purple': 'Purple',
    'Bronze/Copper': 'Other',
    'Other Yellow': 'Yellow',
    'Other Turquoise': 'Turquoise',
    'Other Green': 'Green',
    'Other Blue': 'Blue'    }   
    recommended_items_mapped = pd.DataFrame()
    recommended_items_mapped['mapped_color'] = recommended_items['colour_group_name'].map(color_mapping).fillna(recommended_items['colour_group_name'])

    return recommended_items_mapped,color_mapping
def get_lighter_color(color):
    lighter_color_mapping = {
        "Black": "#aaaaaa",
        "White": "#eeeeee",
        "Beige": "#F5F5DC",
        "Grey": "#A9A9A9",
        "Light Blue": "#399ba3",
        "Pink": "#FF69B4",
        "Red": "#bd3939",
        "Orange": "#FFA500",
        "Yellow": "#FFFF00",
        "Green": "#9addbd",
        "Purple": "#800080",
        "Turquoise": "#40E0D0",
        "Other": "#FFFFFF"
    }
    return lighter_color_mapping.get(color, color)  

#plt functions that use matplotlib
def plt_index_grp_name(recommended_items):
    grouped_data = recommended_items.groupby('index_group_name').size()

    # Plotting the graph (bar plot in this case)
    plt.figure(figsize=(10, 6))  # Define the size of the plot

    grouped_data.plot(kind='bar', color='skyblue')  # Create a bar plot
    plt.title('Counts by Index Group Name')  # Add a title to the plot
    plt.xlabel('Index Group Name')  # Label for x-axis
    plt.ylabel('Count')  # Label for y-axis
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
    plt.grid(axis='y')  # Show gridlines on y-axis

    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    return plt

def plt_color_grp(recommended_items):
    #creo un dataframe nuevo con los colores mapeados
    recommended_items,color_map = color_mapping(recommended_items)
    #los agrupo
    grouped_data = recommended_items.groupby('mapped_color').size()
    #asi los ordenamos
    grouped_data = grouped_data.sort_values(ascending=False)
    #hago que cada color sea mas bonito
    colors_to_plot = list(color_map.values())
    lighter_color_print_map = [get_lighter_color(color) for color in colors_to_plot]

    grouped_data.plot(kind='bar', color = lighter_color_print_map) 
    plt.title('Counts by Colour Group Name') 
    plt.xlabel('Colour Group Name')  
    plt.ylabel('Count') 
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
    plt.grid(False)

    
    plt.gca().set_facecolor((0.9, 0.9, 0.9))  # Light grey using RGB values
    plt.tight_layout() 
    return plt

#sns functions that use seaborn
def sns_color_grp2(recommended_items):
    sns.set_theme(style="whitegrid")
    sns.set_palette("pastel")
    recommended_items,color_map = color_mapping(recommended_items)
    #los agrupo
    grouped_data = recommended_items.groupby('mapped_color').size()
    #asi los ordenamos
    grouped_data = grouped_data.sort_values(ascending=False)
    # Create the bar plot using Seaborn
    ax = sns.barplot(x='Colour Group Name', y='Count', data=grouped_data)
    ax.set_title('Counts by Colour Group Name')
    ax.set_xlabel('Colour Group Name')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # Rotate x-axis labels for better readability if needed

    # Adjust plot aesthetics
    sns.despine(left=True)  # Remove the left spine
    plt.tight_layout()