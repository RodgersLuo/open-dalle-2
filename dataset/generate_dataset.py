from abc import abstractmethod
import numpy as np
import numpy.random as random
# from skimage import draw
import PIL
from PIL import Image
from PIL.ImageDraw import ImageDraw
import pandas as pd
import os

COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "silver": (192, 192, 192),
    "gray": (128, 128, 128),
    "maroon": (128, 0, 0),
    "olive": (128, 128, 0),
    "green": (0, 128, 0),
    "purple": (128, 0, 128),
    "teal": (0, 128, 128),
    "navy": (0, 0, 128),
    "orange": (255, 165, 0),
    "gold": (255, 215, 0),
    "pink": (255, 192, 203),
}

SHAPES = [
    "rectangle",
    "square",
    "ellipse",
    "circle",
    "triangle",
    "pentagon",
    # "hexagon",
    # "heptagon",
]

SIZES = [
    "small",
    "medium",
    "large",
]


class Drawing(object):

    def __init__(self, imagesize, background_color, excluded=None):
        """
        :param imagesize: size of the image in the form (rows, cols)
        :param background_color: color of the background
        :param excluded: list of colors and shapes to exclude
        """
        self.imagesize = imagesize
        self.background_color = background_color
        self.image = Image.new("RGB", imagesize, color=COLORS[background_color])
        self.draw = PIL.ImageDraw.Draw(self.image)
        self.figures = []
        self.excluded = excluded

    def add_figure(self, color, shape, min_length=5):
        """
        Add a figure to the image
        :param color: color of the figure
        :param shape: shape of the figure
        """
        def create_figure():
            height, width = self.imagesize
            x1 = random.randint(0, width-min_length)
            x2 = random.randint(x1+min_length, width)
            y1 = random.randint(0, height-min_length)
            y2 = random.randint(y1+min_length, height)
            r = min((x2 - x1) // 2, (y2 - y1) // 2)
            xc = (x1 + x2) // 2
            yc = (y1 + y2) // 2
            rotation = random.randint(0, 360)

            if shape == "rectangle":
                figure = Rectangle(x1, y1, x2, y2, color)
            elif shape == "square":
                x2 = min(x2, x1 + (y2 - y1))
                figure = Square(x1, y1, x2, color)
            elif shape == "ellipse":
                figure = Ellipse(x1, y1, x2, y2, color)
            elif shape == "circle":
                x2 = min(x2, x1 + (y2 - y1))
                figure = Circle(x1, y1, x2, color)
            elif shape == "triangle":
                figure = Polygon(xc, yc, r, color, n_sides=3, rotation=rotation)
            elif shape == "pentagon":
                figure = Polygon(xc, yc, r, color, n_sides=5, rotation=rotation)
            elif shape == "hexagon":
                figure = Polygon(xc, yc, r, color, n_sides=6, rotation=rotation)
            elif shape == "heptagon":
                figure = Polygon(xc, yc, r, color, n_sides=7, rotation=rotation)
            else:
                raise ValueError("Shape not recognized")
            return figure

        # check if the figure is excluded
        if self._is_excluded(color, shape):
            return False
        figure = create_figure()
        # check if the figure overlaps with any other figure
        overlapping = False
        for f in self.figures:
            if Figure.overlapping(f, figure):
                overlapping = True
                break
        # add the figure if it does not overlap
        if not overlapping:
            self.figures.append(figure)
            figure.draw(self.draw)
            return True
        else:
            return False

    def add_random_figures(self, max_num, min_length=5):
        n = np.random.randint(1, max_num+1)
        for _ in range(n):
            color = np.random.choice(list(COLORS.keys()))
            shape = np.random.choice(SHAPES)
            # # check if the figure is excluded
            # while not self._is_excluded(color, shape):
            #     color = np.random.choice(list(COLORS.keys()))
            #     shape = np.random.choice(SHAPES)

            # add the figure
            i = 0
            while i < 100 and not self.add_figure(color, shape, min_length=min_length):
                color = np.random.choice(list(COLORS.keys()))
                shape = np.random.choice(SHAPES)
                i += 1

    def caption(self, size_drop_prob, pos_drop_prob):
        captions = [figure.generate_caption(self.imagesize, size_drop_prob=size_drop_prob, pos_drop_prob=pos_drop_prob)
                     for figure in self.figures]
        return " and ".join(captions)

    def show(self):
        self.image.show()

    def save(self, filename):
        self.image.save(filename)

    def _is_excluded(self, color, shape):
        if self.excluded is None:
            return False
        for exc in self.excluded:
            if isinstance(exc, str):
                if exc == color or exc == shape:
                    return True
            if exc[0] == color and exc[1] == shape:
                return True
        return False


class Figure(object):

    def __init__(self, color, pos):
        self.color = color  # name
        self.pos = pos  # as simple bounding box in the form ((r0, r1), (c0, c1))

    @abstractmethod
    def shape(self):
        """
        Return the shape of the figure
        :return: shape
        """
        pass

    @abstractmethod
    def draw(self, draw: ImageDraw):
        """
        Draw the figure on an image
        :param image: image array
        """

    def position_label(self, imagesize):
        """
        Modified from https://github.com/MdeBoer95/random-shapes-with-captions.

        Return the position label based on the coordinates of the bounding box of the shape. The image is devided
        into 4 areas: top left, top right, bottom left, bottom right
        :param imagesize: rows and cols of the image
        :return: a label for the position
        """
        # rows and cols of the image
        rows = imagesize[0]
        cols = imagesize[1]

        # determine middle row and col
        mid_row = int(np.floor(rows / 2))
        mid_col = int(np.floor(cols / 2))

        # Check which box the shape belongs to and assign a label
        horizontal_pos = ""
        vertical_pos = ""

        bounding_box = self.pos
        if bounding_box[0][0] <= mid_row and bounding_box[0][1] <= mid_row:
            vertical_pos = "top"
        elif bounding_box[0][0] > mid_row and bounding_box[0][1] > mid_row:
            vertical_pos = "bottom"

        if bounding_box[1][0] <= mid_col and bounding_box[1][1] <= mid_col:
            horizontal_pos = "left"
        elif bounding_box[1][0] > mid_col and bounding_box[1][1] > mid_col:
            horizontal_pos = "right"

        if horizontal_pos == "" and vertical_pos == "":
            return "center"

        return " ".join([vertical_pos, horizontal_pos]).strip()

    def size_label(self, imagesize):
        """
        Return the size label based on the area of the shape
        :param imagesize: rows and cols of the image
        :return: a label for the size
        """
        image_area = imagesize[0] * imagesize[1]
        shape_area = (self.pos[0][1] - self.pos[0][0]) * (self.pos[1][1] - self.pos[1][0])
        shape_area = np.abs(shape_area)
        ratio = shape_area / image_area
        if ratio < 0.05:
            return "small"
        elif ratio < 0.15:
            return "medium"
        else:
            return "large"

    def generate_caption(self, imagesize, size_drop_prob, pos_drop_prob):
        def assemble_determiner(caption):
            if random.rand() < 0.2:
                return "the " + caption
            else:
                if caption.lower().strip()[0] in "aeiou":
                    return "an " + caption
                else:
                    return "a " + caption

        size = self.size_label(imagesize)
        shape_name = self.shape()
        color_name = self.color
        pos = self.position_label(imagesize)
        if random.rand() < size_drop_prob:
            caption = f"{color_name} {shape_name}"
        else:
            caption = f"{size} {color_name} {shape_name}"

        if random.rand() > pos_drop_prob:
            caption += f" at the {pos}"
        return assemble_determiner(caption)


    @classmethod
    def overlapping(cls, fig1, fig2):
        """
        Modified from https://github.com/MdeBoer95/random-shapes-with-captions.

        Check if two figures overlap according to their bounding boxes (2D space)
        :param fig1: the first figure
        :param fig2: the second figure
        :return: True if the bounding boxes of the figures overlap, False if not
        """

        def overlapping1D(box1, box2):
            (min1, max1) = box1
            (min2, max2) = box2
            return max1 >= min2 and max2 >= min1

        if overlapping1D(fig1.pos[0], fig2.pos[0]) and overlapping1D(fig1.pos[1], fig2.pos[1]):
            return True
        else:
            return False


class Rectangle(Figure):

    def __init__(self, x1, y1, x2, y2, color):
        pos = ((y1, y2), (x1, x2))
        super().__init__(color, pos)

    def shape(self):
        return "rectangle"

    def draw(self, draw: ImageDraw):
        ((y1, y2), (x1, x2)) = self.pos
        draw.rectangle((x1, y1, x2, y2), fill=COLORS[self.color])


class Square(Rectangle):

    def __init__(self, x1, y1, x2, color):
        y2 = y1 + (x2 - x1)
        super().__init__(x1, y1, x2, y2, color)

    def shape(self):
        return "square"


class Ellipse(Figure):
    def __init__(self, x1, y1, x2, y2, color):
        pos = ((y1, y2), (x1, x2))
        super().__init__(color, pos)

    def shape(self):
        return "ellipse"

    def draw(self, draw: ImageDraw):
        ((y1, y2), (x1, x2)) = self.pos
        draw.ellipse((x1, y1, x2, y2), fill=COLORS[self.color])


class Circle(Ellipse):
    def __init__(self, x1, y1, x2, color):
        y2 = y1 + (x2 - x1)
        super().__init__(x1, y1, x2, y2, color)

    def shape(self):
        return "circle"


class Polygon(Figure):
    def __init__(self, x, y, r, color, n_sides, rotation=0):
        assert n_sides in [3, 5, 6, 7]
        x1, x2 = x - r, x + r
        y1, y2 = y - r, y + r
        pos = ((y1, y2), (x1, x2))
        super().__init__(color, pos)
        self.n_sides = n_sides
        self.rotation = rotation
        self.x = x
        self.y = y
        self.r = r

    def shape(self):
        if self.n_sides == 3:
            return "triangle"
        elif self.n_sides == 5:
            return "pentagon"
        elif self.n_sides == 6:
            return "hexagon"
        elif self.n_sides == 7:
            return "heptagon"

    def draw(self, draw: ImageDraw):
        draw.regular_polygon((self.x, self.y, self.r), self.n_sides, rotation=self.rotation, fill=COLORS[self.color])


if __name__ == "__main__":
    n_test = 2000
    n_train = 20000
    train_excluded = [
        "gray",
        ("white", "rectangle"),
        ("gold", "triangle"),
        ("blue", "circle"),
        ("red", "pentagon"),
        ("green", "square"),
        ("yellow", "ellipse"),
    ]

    test_excluded = [
        "gray",
    ]

    train_dir = "./data/train"
    # test_dir = "./data/test"
    test_dir = "./evaluations/v3/reference"

    # os.makedirs(train_dir + "/images")
    os.makedirs(test_dir + "/images")

    size_drop_prob = 0.3
    pos_drop_prob = 1

    # captions, filenames = [], []
    # for i in range(n_train):
    #     filename = f"train_{i}.png"
    #     drawing = Drawing((128, 128), background_color="gray", excluded=train_excluded)
    #     # if i < n_train / 4:
    #     #     drawing.add_random_figures(1, min_length=20)
    #     # elif i < 2 * n_train / 4:
    #     #     drawing.add_random_figures(2, min_length=20)
    #     # else:
    #     #     drawing.add_random_figures(3, min_length=20)
    #     drawing.add_random_figures(2, min_length=20)
    #     captions.append(drawing.caption(size_drop_prob=size_drop_prob, pos_drop_prob=pos_drop_prob))
    #     filenames.append(filename)
    #     drawing.save(os.path.join(train_dir, "images", filename))

    #     if i % 1000 == 0:
    #         print(f"Generated {i} images")

    # train_df = pd.DataFrame({"caption": captions, "image": filenames})
    # train_df.to_csv(train_dir + "/data.csv")

    # captions, filenames = [], []
    # for i in range(n_test):
    #     filename = f"test_{i}.png"
    #     drawing = Drawing((128, 128), background_color="gray", excluded=test_excluded)
    #     # if i < n_test / 4:
    #     #     drawing.add_random_figures(1, min_length=20)
    #     # elif i < 2 * n_test / 4:
    #     #     drawing.add_random_figures(2, min_length=20)
    #     # else:
    #     #     drawing.add_random_figures(3, min_length=20)
    #     drawing.add_random_figures(2, min_length=20)
    #     captions.append(drawing.caption(size_drop_prob=size_drop_prob, pos_drop_prob=pos_drop_prob))
    #     filenames.append(filename)
    #     drawing.save(os.path.join(test_dir, "images", filename))

    #     if i % 1000 == 0:
    #         print(f"Generated {i} images")

    # test_df = pd.DataFrame({"caption": captions, "image": filenames})
    # test_df.to_csv(test_dir + "/data.csv")


    captions, filenames = [], []
    for i in range(1000):
        filename = f"{i}.png"
        drawing = Drawing((128, 128), background_color="gray")
        # if i < n_test / 4:
        #     drawing.add_random_figures(1, min_length=20)
        # elif i < 2 * n_test / 4:
        #     drawing.add_random_figures(2, min_length=20)
        # else:
        #     drawing.add_random_figures(3, min_length=20)
        i = np.random.randint(1, len(train_excluded))
        caption = train_excluded[i]
        drawing.add_figure(caption[0], caption[1], min_length=20)
        captions.append(drawing.caption(size_drop_prob=size_drop_prob, pos_drop_prob=pos_drop_prob))
        filenames.append(filename)
        drawing.save(os.path.join(test_dir, "images", filename))

        if i % 100 == 0:
            print(f"Generated {i} images")

    test_df = pd.DataFrame({"caption": captions, "image": filenames})
    test_df.to_csv(test_dir + "/data.csv")
