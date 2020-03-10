#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang 
# E-mail     : liang@informatik.uni-hamburg.de
# Description: 
# Date       : 15/03/2019: 11:03 AM
# File Name  : generate_bottle_config
from __future__ import print_function, division
import numpy as np
import os
from audio_pouring.utils.utils import config, pkg_path, poly_fit_with_fixed_points
from numpy import genfromtxt
import matplotlib.pyplot as plt
import seaborn as sns


def bottle_config(update_npy=True, vis=True):
    bottle_id_list = config["bottle_id_list"]
    for bottle_id in bottle_id_list:
        bottle_raw_data_path = os.path.join(pkg_path, "config/bottles", "bottle" + str(bottle_id) + "_config.csv")
        input_bottle_data = genfromtxt(bottle_raw_data_path, delimiter=",", skip_header=1)
        x = input_bottle_data[:, 1] / 1000.0
        y = input_bottle_data[:, 0]
        xf = np.array([input_bottle_data[0, 1] / 1000.0, input_bottle_data[-1, 1] / 1000.0])
        yf = np.array([input_bottle_data[0, 0], input_bottle_data[-1, 0]])
        params = poly_fit_with_fixed_points(degree=2, x=x, y=y, x_fix=xf, y_fix=yf)
        print("Bottle {} params are: {}".format(str(bottle_id), params))

        poly = np.polynomial.Polynomial(params)
        draw_line = np.linspace(x[0], x[-1], 50)

        if vis:
            sns.set(palette="deep", color_codes=True)
            with sns.axes_style("darkgrid"):
                fig = plt.figure()
                fig.set_size_inches(6, 6)
                plt.plot(x, y, "bo", label="measured float point")
                plt.plot(xf, yf, "ro", label="measured fixed point")
                plt.plot(draw_line, poly(draw_line), "-", label="fitting curve")
                # plt.title("Bottle {} data fitting".format(str(bottle_id)), fontsize=16)
                plt.xlabel("Weight (kg)", fontsize=16)
                plt.ylabel("Length of air column (mm)", fontsize=16)
                legend_size = 18
                plt.legend(loc=1, prop={"size": legend_size})
                plt.savefig("/tmp/weight_to_height.pdf")
                plt.show()
        if update_npy:
            np.save(bottle_raw_data_path[:-4] + ".npy", params)


if __name__ == "__main__":
    bottle_config(vis=False)
