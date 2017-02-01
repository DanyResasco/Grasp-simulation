#!/usr/bin/env python

import pkg_resources
pkg_resources.require("klampt>=0.7.0")
from klampt import *
from klampt import vis 
from klampt.vis.glrobotprogram import *
from klampt.math import *
from klampt.model import collide
from klampt.io import resource
from klampt.sim import *
from moving_base_control import *
import importlib
import math
import os
import string
import sys
import time
import pickle
import csv
from create_mvbb import MVBBVisualizer, compute_poses, skip_decimate_or_return
from create_mvbb_filtered import FilteredMVBBVisualizer
from klampt.math import so3, se3
import pydany_bb
import numpy as np
from IPython import embed
from mvbb.graspvariation import PoseVariation
from mvbb.TakePoses import SimulationPoses
from mvbb.draw_bbox import draw_GL_frame, draw_bbox
from i16mc import make_object, make_moving_base_robot
from mvbb.CollisionCheck import CheckCollision, CollisionTestInterpolate, CollisionTestPose, CollisionCheckWordFinger
from mvbb.db import MVBBLoader
from mvbb.kindness import Differential,RelativePosition
from mvbb.GetForces import get_contact_forces_and_jacobians
from mvbb.ScalaReduce import DanyReduceScale
from dany_make_rotate_voxel import make_objectRotate



'''Simulation of rotation mesh'''




objects = {}
objects['ycb'] = [f for f in os.listdir('data/objects/voxelrotate/ycb')]
objects['apc2015'] = [f for f in os.listdir('data/objects/voxelrotate/apc2015')]
# objects['newObjdany'] = [f for f in os.listdir('data/objects/voxelrotate/newObjdany')]
objects['princeton'] = [f for f in os.listdir('data/objects/voxelrotate/princeton')]
robots = ['reflex_col']

done = []

# done = ['morton_salt_shaker_rotate_1','champion_sports_official_softball_rotate_1',
#     'melissa_doug_farm_fresh_fruit_banana_rotate_1','champion_sports_official_softball_rotate_1',]
# done = ['champion_sports_official_softball_rotate_1','black_and_decker_lithium_drill_driver_unboxed_rotate_3','black_and_decker_lithium_drill_driver_unboxed_rotate_16',
# 'black_and_decker_lithium_drill_driver_unboxed_rotate_18','black_and_decker_lithium_drill_driver_unboxed_rotate_32','black_and_decker_lithium_drill_driver_unboxed_rotate_33','soft_scrub_2lb_4oz_rotate_5',
# 'soft_scrub_2lb_4oz_rotate_6','soft_scrub_2lb_4oz_rotate_13','learning_resources_one-inch_color_cubes_box_rotate_2','learning_resources_one-inch_color_cubes_box_rotate_3',
# 'learning_resources_one-inch_color_cubes_box_rotate_4','learning_resources_one-inch_color_cubes_box_rotate_5','stainless_steel_spoon_red_handle_rotate_2','stainless_steel_spoon_red_handle_rotate_4',
# 'stainless_steel_spoon_red_handle_rotate_5','stainless_steel_spoon_red_handle_rotate_6','stainless_steel_spoon_red_handle_rotate_7','stainless_steel_spoon_red_handle_rotate_8','stainless_steel_fork_red_handle_rotate_1',
# 'stainless_steel_fork_red_handle_rotate_2','champion_sports_official_softball_rotate_2','black_and_decker_lithium_drill_driver_unboxed_rotate_6','stainless_steel_fork_red_handle_rotate_3','stainless_steel_fork_red_handle_rotate_4',
# 'stainless_steel_fork_red_handle_rotate_5','stainless_steel_fork_red_handle_rotate_6','stainless_steel_fork_red_handle_rotate_7','stainless_steel_fork_red_handle_rotate_8','stainless_steel_fork_red_handle_rotate_9',
# 'stainless_steel_fork_red_handle_rotate_10','stainless_steel_fork_red_handle_rotate_11','stainless_steel_fork_red_handle_rotate_12','champion_sports_official_softball_rotate_3','black_and_decker_lithium_drill_driver_unboxed_rotate_12',
# 'black_and_decker_lithium_drill_driver_unboxed_rotate_41','stainless_steel_fork_red_handle_rotate_13','stainless_steel_fork_red_handle_rotate_16','stainless_steel_fork_red_handle_rotate_17','stainless_steel_fork_red_handle_rotate_18',
# 'stainless_steel_fork_red_handle_rotate_19','stainless_steel_fork_red_handle_rotate_21','stainless_steel_fork_red_handle_rotate_24','stainless_steel_fork_red_handle_rotate_27','stainless_steel_fork_red_handle_rotate_28',
# 'stainless_steel_fork_red_handle_rotate_30','wescott_orange_grey_scissors_rotate_4','wescott_orange_grey_scissors_rotate_5','wescott_orange_grey_scissors_rotate_6','wescott_orange_grey_scissors_rotate_9','wescott_orange_grey_scissors_rotate_14',
# 'wescott_orange_grey_scissors_rotate_15','wescott_orange_grey_scissors_rotate_18','wescott_orange_grey_scissors_rotate_19','wescott_orange_grey_scissors_rotate_22','wescott_orange_grey_scissors_rotate_27','wescott_orange_grey_scissors_rotate_30',
# 'wescott_orange_grey_scissors_rotate_31','wescott_orange_grey_scissors_rotate_33','large_black_spring_clamp_rotate_8','large_black_spring_clamp_rotate_24','large_black_spring_clamp_rotate_29','large_black_spring_clamp_rotate_30','play_go_rainbow_stakin_cups_box_rotate_35',
# 'play_go_rainbow_stakin_cups_box_rotate_36','frenchs_classic_yellow_mustard_14oz_rotate_14','windex_rotate_14','windex_rotate_19','stainless_steel_knife_red_handle_rotate_3','stainless_steel_knife_red_handle_rotate_4','stainless_steel_knife_red_handle_rotate_6',
# 'frenchs_classic_yellow_mustard_14oz_rotate_1','champion_sports_official_softball_rotate_4','black_and_decker_lithium_drill_driver_unboxed_rotate_19','frenchs_classic_yellow_mustard_14oz_rotate_15','frenchs_classic_yellow_mustard_14oz_rotate_18','frenchs_classic_yellow_mustard_14oz_rotate_21',
# 'windex_rotate_7','stainless_steel_knife_red_handle_rotate_7','stainless_steel_knife_red_handle_rotate_8','stainless_steel_knife_red_handle_rotate_13','stainless_steel_knife_red_handle_rotate_17','champion_sports_official_softball_rotate_5',
# 'champion_sports_official_softball_rotate_6','black_and_decker_lithium_drill_driver_unboxed_rotate_25','learning_resources_one-inch_color_cubes_box_rotate_1','play_go_rainbow_stakin_cups_box_rotate_33','medium_black_spring_clamp_rotate_1'
# 'stainless_steel_knife_red_handle_rotate_18','stainless_steel_knife_red_handle_rotate_21','stainless_steel_knife_red_handle_rotate_22','wearever_cooking_pan_with_lid_rotate_8','medium_black_spring_clamp_rotate_9'
# 'wearever_cooking_pan_with_lid_rotate_9','wearever_cooking_pan_with_lid_rotate_10','wearever_cooking_pan_with_lid_rotate_11','wearever_cooking_pan_with_lid_rotate_12','champion_sports_official_softball_rotate_7','black_and_decker_lithium_drill_driver_unboxed_rotate_11',
# 'black_and_decker_lithium_drill_driver_unboxed_rotate_42','champion_sports_official_softball_rotate_8','black_and_decker_lithium_drill_driver_unboxed_rotate_38','soft_scrub_2lb_4oz_rotate_3','wearever_cooking_pan_with_lid_rotate_13','wearever_cooking_pan_with_lid_rotate_14',
# 'wearever_cooking_pan_with_lid_rotate_15','wearever_cooking_pan_with_lid_rotate_16','wearever_cooking_pan_with_lid_rotate_17','wearever_cooking_pan_with_lid_rotate_19','wearever_cooking_pan_with_lid_rotate_20',
# 'wearever_cooking_pan_with_lid_rotate_22','wearever_cooking_pan_with_lid_rotate_25','wearever_cooking_pan_with_lid_rotate_26','wearever_cooking_pan_with_lid_rotate_31','wearever_cooking_pan_with_lid_rotate_32',
# 'wearever_cooking_pan_with_lid_rotate_35','wearever_cooking_pan_with_lid_rotate_36','pringles_original_rotate_3','pringles_original_rotate_4','champion_sports_official_softball_rotate_9','pringles_original_rotate_5','black_and_decker_lithium_drill_driver_unboxed_rotate_24',
# 'black_and_decker_lithium_drill_driver_unboxed_rotate_49','black_and_decker_lithium_drill_driver_unboxed_rotate_49','master_chef_ground_coffee_297g_rotate_2','large_black_spring_clamp_rotate_14','play_go_rainbow_stakin_cups_3_red_rotate_21',
# 'rubbermaid_ice_guard_pitcher_blue_rotate_20','rubbermaid_ice_guard_pitcher_blue_rotate_22','rubbermaid_ice_guard_pitcher_blue_rotate_23','rubbermaid_ice_guard_pitcher_blue_rotate_24','stanley_13oz_hammer_rotate_4',
# 'master_chef_ground_coffee_297g_rotate_6','black_and_decker_lithium_drill_driver_unboxed_rotate_7','stanley_13oz_hammer_rotate_5','first_years_take_and_toss_straw_cups_rotate_5','first_years_take_and_toss_straw_cups_rotate_14',
# 'first_years_take_and_toss_straw_cups_rotate_23','first_years_take_and_toss_straw_cups_rotate_38','first_years_take_and_toss_straw_cups_rotate_37','allcolorcar_rotate_1','allcolorcar_rotate_2','azureship_rotate_1','azureship_rotate_2',
# 'azureship_rotate_3','azureship_rotate_4','backoldcarriage_rotate_1','backoldcarriage_rotate_2','backoldcarriage_rotate_3','backoldcarriage_rotate_4','backoldcarriage_rotate_5','backoldcarriage_rotate_6','mead_index_cards_rotate_1',
# 'batmancar_rotate_1','batmancar_rotate_2','batmancar_rotate_3','behindbluecar_rotate_1','behindbluecar_rotate_15','behindblueformula1_rotate_2','behindformula1_rotate_1','behindtruck_rotate_1','behindwhitecar_rotate_1',
# 'bigblueskate_rotate_1','bigparabola_rotate_1','bigparabola_rotate_2','bishop_rotate_1','bishop_rotate_10','blackandgreenship_rotate_10','blackandstripeshat_rotate_2,','blackandwhitetank_rotate_1','blackandwhitetank_rotate_9',
# 'blackandwhitetank_rotate_10','blackandwhitetank_rotate_11','blackandwhitetank_rotate_12','blackandwhitetank_rotate_13','blackandwhitetank_rotate_14','blackandwhitetank_rotate_15','blackbook_rotate_1','blackcatamaran_rotate_1','blackcatamaran_rotate_3',
# 'blackformula1_rotate_1','blackoldhat_rotate_1','blackoldhat_rotate_2','blackshoes_rotate_1','blackshoes_rotate_2','blackshoes_rotate_3','blackshoes_rotate_4','blackshoes_rotate_5','blackshoes_rotate_6','blackshoes_rotate_7',
# 'blacktank_rotate_1','blacktank_rotate_5','blacktruck_rotate_1','blacktruck_rotate_2','block_of_wood_6in_rotate_1','block_of_wood_6in_rotate_1','block_of_wood_6in_rotate_5','blueandredbigcar_rotate_3','bluecar_rotate_3','bluemovingballs_rotate_4','bluemovingballs_rotate_5','bluemovingballs_rotate_6',
# 'brine_mini_soccer_ball_rotate_1','bronwcowboyhat_rotate_1','bronwcowboyhat_rotate_2','bronwstair_rotate_3','bronwstair_rotate_4','brownchess_rotate_1','brownchess_rotate_2','brownchess_rotate_3',
# 'browndragonwithwings_rotate_1','browndragonwithwings_rotate_2','browndragonwithwings_rotate_3','browndragonwithwings_rotate_4','browndragonwithwings_rotate_5','browndragonwithwings_rotate_6','browndragonwithwings_rotate_7','browndragonwithwings_rotate_9',
# 'brownfireplace_rotate_1','brownfireplace_rotate_3','captainhat_rotate_3','captainhat_rotate_4','car_rotate_1','car_rotate_2','car_rotate_3','car_rotate_4','carwithflag_rotate_1','carwithflag_rotate_2','carwithflag_rotate_3','carwithflag_rotate_4',
# 'cashmachine_rotate_1','cashmachinebig_rotate_1','cashmachineblack_rotate_1','cashmachineblack_rotate_2','cheezit_big_original_rotate_1','cheezit_big_original_rotate_1','cheezit_big_original_rotate_2','chess_rotate_10',
# 'clorox_disinfecting_wipes_35_rotate_1','chess_rotate_11','chess_rotate_12','chess_rotate_13','chess_rotate_14','chinesbridge_rotate_1','chinesbridge_rotate_2','chinesbridge_rotate_12','colorcar2_rotate_1','comet_lemon_fresh_bleach_rotate_1','crayola_64_ct_rotate_1','crayola_64_ct_rotate_7',
# 'crayola_64_ct_rotate_14','dark_red_foam_block_with_three_holes_rotate_1','domino_sugar_1lb_rotate_1','doublegraydoor_rotate_1','dr_browns_bottle_brush_rotate_1','dr_browns_bottle_brush_rotate_5','dr_browns_bottle_brush_rotate_6','dr_browns_bottle_brush_rotate_7',
# 'elmers_washable_no_run_school_glue_rotate_1','etruscansailboat_rotate_1','etruscansailboat_rotate_2','etruscansailboat_rotate_3','expo_dry_erase_board_eraser_rotate_1','feline_greenies_dental_treats_rotate_1','mark_twain_huckleberry_finn_rotate_1',
# 'feline_greenies_dental_treats_rotate_4','feline_greenies_dental_treats_rotate_14','first_years_take_and_toss_straw_cups_rotate_1','flipoverblackship_rotate_11','flipoverblackship_rotate_13','flipoverblackship_rotate_19','master_chef_ground_coffee_297g_rotate_1',
# 'flipovergreentank_rotate_1','flipoveropencar_rotate_1','flipoveropencar_rotate_7','flipoverwhitecar_rotate_4','flipoverwhitemoto_rotate_1','flipoveryellowskate_rotate_1','flipoveryellowskate_rotate_2','flipoveryellowskate_rotate_3',
# 'flipoveryellowskate_rotate_4','flipoveryellowskate_rotate_5','flipoveryellowskate_rotate_6','flipoveryellowskate_rotate_7','flipoveryellowskate_rotate_8','flipoveryellowskate_rotate_9','flipoveryellowskate_rotate_10','flipoveryellowskate_rotate_11',
# 'flipoveryellowskate_rotate_12','flipoveryellowskate_rotate_13','flipoveryellowskate_rotate_14','flipoveryellowskate_rotate_15','flipoveryellowskate_rotate_16','flipoveryellowskate_rotate_17','flipoveryellowskate_rotate_18','flipoveryellowskate_rotate_19','flipoveryellowskate_rotate_20',
# 'formula1_rotate_1','formula1_rotate_6','formula1_rotate_7','formula1_rotate_8','formula1_rotate_13','formula1_rotate_14','formula1_rotate_17','frenchs_classic_yellow_mustard_14oz_rotate_2','genuine_joe_plastic_stir_sticks_rotate_1',
# 'graybook_rotate_1','graystripesshoes_rotate_1','graystripesshoes_rotate_2','grayubot_rotate_1','grayubot_rotate_2','greendragonwithwings_rotate_1','greendragonwithwings_rotate_5','greendragonwithwings_rotate_8','greendragonwithwings_rotate_9','greendragonwithwings_rotate_10',
# 'greendragonwithwings_rotate_12','greendragonwithwings_rotate_13','greendragonwithwings_rotate_14','greendragonwithwings_rotate_15','greendragonwithwings_rotate_16','greendragonwithwings_rotate_17','greendragonwithwings_rotate_25','greendragonwithwings_rotate_26',
# 'greendragonwithwings_rotate_27','greendragonwithwings_rotate_29','greendragonwithwings_rotate_30','greenskate_rotate_1','greenskate_rotate_4','greenskate_rotate_6','greentank_rotate_1','greentank_rotate_2','greentank_rotate_3','harley_rotate_1',
# 'highhatwhite_rotate_1','highhatwhite_rotate_14','highland_6539_self_stick_notes_rotate_1','historyhat_rotate_1','historyhat_rotate_2','o_chocolate_flavor_pudding_rotate_1','king2_rotate_1','locomotive_rotate_2','locomotive_rotate_3','locomotive_rotate_5'
# 'king2_rotate_4','king_rotate_1','kong_air_dog_squeakair_tennis_ball_rotate_1','kong_air_dog_squeakair_tennis_ball_rotate_6','kong_air_dog_squeakair_tennis_ball_rotate_7','kong_duck_dog_toy_rotate_1','kong_sitting_frog_dog_toy_rotate_1','locomotive_rotate_6',
# 'learning_resources_one-inch_color_cubes_box_rotate_6','kygen_squeakin_eggs_plush_puppies_rotate_1','large_black_spring_clamp_rotate_1','large_black_spring_clamp_rotate_13','longblacksatellite_rotate_3','longpinkcar_rotate_1','longpinkcar_rotate_7',
# 'longpinkcar_rotate_8','longpinkcar_rotate_22','longpinkcar_rotate_33','longpinksatellite_rotate_1','longredcontainership_rotate_6','longredcontainership_rotate_5','longwhitecontainership_rotate_1','lyingblackshoes_rotate_1','lyingblackshoes_rotate_6','lyingblackshoes_rotate_7',
# 'lyingblackshoes_rotate_15','lyingoldcarriage_rotate_1','lyingoldcarriage_rotate_3','lyingoldcarriage_rotate_5','lyingoldcarriage_rotate_6','lyingoldcarriage_rotate_7','lyingoldcarriage_rotate_8','lyingoldcarriage_rotate_9','lyingoldcarriage_rotate_10','lyingoldcarriage_rotate_11',
# 'lyingoldcarriage_rotate_18','lyingoldcarriage_rotate_19','lyingoldcarriage_rotate_21','lyingoldcarriage_rotate_27','lyingstaircase_rotate_2','lyingstaircase_rotate_3','lyingstaircase_rotate_4','manyblock_rotate_1','manyblock_rotate_2',
# 'melissa_doug_farm_fresh_fruit_apple_rotate_1','melissa_doug_farm_fresh_fruit_banana_rotate_1','melissa_doug_farm_fresh_fruit_lemon_rotate_1','melissa_doug_farm_fresh_fruit_orange_rotate_1','melissa_doug_farm_fresh_fruit_peach_rotate_1',
# 'melissa_doug_farm_fresh_fruit_plum_rotate_1','melissa_doug_farm_fresh_fruit_strawberry_rotate_1','militarygreencar_rotate_2','militarygreencar_rotate_3','militarygreencar_rotate_4','militarygreencar_rotate_5','militarygreencar_rotate_6',
# 'militaryhat_rotate_4', 'militarygreencar_rotate_7','militarygreencar_rotate_8','militaryhat_rotate_5','militaryhat_rotate_6','militaryhat_rotate_7','militaryship_rotate_1','militaryship_rotate_2',
# 'miniflagcar_rotate_1','miniflagcar_rotate_3','miniflagcar_rotate_4','miniflagcar_rotate_5','miniflagcar_rotate_6','mommys_helper_outlet_plugs_rotate_1','morton_salt_shaker_rotate_1',
# 'munchkin_white_hot_duck_bath_toy_rotate_1','newbalanceshoes_rotate_16','newbalanceshoes_rotate_17','oldcar_rotate_2','oldcar_rotate_3','oldcar_rotate_4','oldcar_rotate_5','oldcar_rotate_6',
# 'oldpc_rotate_1','oldpc_rotate_10','oldwhitehat_rotate_2','oldwhitehat_rotate_1','opencar_rotate_1','opencar_rotate_2','opencar_rotate_11','oldwhitehat_rotate_3','oldwhitehat_rotate_4',
# 'orangeamerica_rotate_1','orangewingboat_rotate_1','paper_mate_12_count_mirado_black_warrior_rotate_1','paper_mate_12_count_mirado_black_warrior_rotate_13','penn_raquet_ball_rotate_1',
# 'pickup_rotate_3','pickup_rotate_4','pickup_rotate_5','pickup_rotate_6','pickup_rotate_7','pickup_rotate_8','pilothat_rotate_2','pilothat_rotate_3','pilothat_rotate_4','pilothat_rotate_5',
# 'pilothat_rotate_6','pilothat_rotate_7','pilothat_rotate_8','pilothat_rotate_9','pilothat_rotate_10','pilothat_rotate_11','pilothat_rotate_12','pilothat_rotate_13',
# 'pinklocomotive_rotate_2','pinklocomotive_rotate_5','pinklocomotive_rotate_6','pinklocomotive_rotate_7','pinklocomotive_rotate_8','pinklocomotive_rotate_9'
# 'play_go_rainbow_stakin_cups_10_blue_rotate_1','pterosauri_rotate_1','purplexwing_rotate_1','purplexwing_rotate_2','purplexwing_rotate_3','purplexwing_rotate_4',
# 'queen_rotate_1','receiver_rotate_1','receiver_rotate_2','receiver_rotate_3','receiver_rotate_4','receiver_rotate_5','receiver_rotate_6','receiver_rotate_7','red_wood_block_1inx1in_rotate_1',
# 'redandblueumbrella_rotate_1','redandblueumbrella_rotate_2','redandblueumbrella_rotate_3','redandgreenbridge_rotate_1','redandwoodstair_rotate_1','redandwoodstair_rotate_2',
# 'redbigcar_rotate_1','redbigcar_rotate_2','redbike_rotate_1','redbike_rotate_2','redformula1_rotate_9','redlocomotive_rotate_1','redmanbike_rotate_1','redmanbike_rotate_2',
# 'redmoto_rotate_1','redmoto_rotate_2','redmoto_rotate_3','redmoto_rotate_4','redmoto_rotate_5','redmoto_rotate_6','redmoto_rotate_7','redmoto_rotate_8','redmoto_rotate_9','redmoto_rotate_10',
# 'redstair_rotate_1','redxwing_rotate_1','redxwing_rotate_2','redxwing_rotate_3','rollodex_mesh_collection_jumbo_pencil_cup_rotate_1','rollodex_mesh_collection_jumbo_pencil_cup_rotate_2','rollodex_mesh_collection_jumbo_pencil_cup_rotate_3',
# 'romansailboat_rotate_2','romansailboat_rotate_3','romansailboat_rotate_4','romansailboat_rotate_6','rubbermaid_ice_guard_pitcher_blue_rotate_1','rubbermaid_ice_guard_pitcher_blue_rotate_10',
# 'safety_works_safety_glasses_rotate_1','sailboat_rotate_1','sideubot_rotate_1','sideubot_rotate_7','sideubot_rotate_8','sideubot_rotate_11','sideubot_rotate_12',
# 'sideubot_rotate_13','sideubot_rotate_14','sideubot_rotate_15','sink_rotate_1','sink_rotate_9','sink_rotate_10','sink_rotate_13','sink_rotate_20','sink_rotate_24',
# 'sink_rotate_25','sink_rotate_26','smallboat_rotate_1','smallboat_rotate_2','smallboat_rotate_3','smallcity2_rotate_1','smallcity3_rotate_1','smallcity3_rotate_7',
# 'smallcity3_rotate_8','smallcity3_rotate_17','smallcity3_rotate_20','smallcity3_rotate_25','smallcitycolor_rotate_1','smallcitycolor_rotate_3','smallcitycolor_rotate_4',
# 'smallwhitetank_rotate_2','smallwhitetank_rotate_3','smallwhitetank_rotate_4','smallwhitetank_rotate_5','smallwhitetank_rotate_6','smallwhitetank_rotate_7','smallwhitetank_rotate_8',
# 'smallwhitetank_rotate_9','snowman2_rotate_1','snowman2_rotate_4','snowman2_rotate_5','snowman2_rotate_6','snowman_rotate_1','snowman_rotate_5','snowmanwhithat_rotate_1','snowmanwhithat_rotate_27',
# 'snowmanwhithatandarm_rotate_1','snowmanwithtrianglehat_rotate_1','snowmanwithtrianglehat_rotate_2','soft_scrub_2lb_4oz_rotate_1','squaresink_rotate_1','squaresink_rotate_2','squaresink_rotate_5',
# 'stainless_steel_fork_red_handle_rotate_14','stainless_steel_spoon_red_handle_rotate_1','stairwhite_rotate_1','stairwhite_rotate_2','stanley_13oz_hammer_rotate_1','standingcar_rotate_1','stanley_66_052_rotate_1',
# 'starkist_chunk_light_tuna_rotate_1','tankblack_rotate_4','tankblack_rotate_5','tankblack_rotate_6','tankblack_rotate_7','texasyellowhat_rotate_2','texasyellowhat_rotate_3',
# 'thick_wood_block_6in_rotate_1','totalblacktank_rotate_5','totalblacktank_rotate_6','totalblacktank_rotate_7','totalblackwithwhitewheeltank_rotate_1','totalblackwithwhitewheeltank_rotate_2',
# 'totalblackwithwhitewheeltank_rotate_3','totalblackwithwhitewheeltank_rotate_4','totalwhitetank_rotate_1','totalwhitetank_rotate_2','totalwhitetank_rotate_3','totalwhitetank_rotate_6',
# 'totalwhitetank_rotate_7','totalwhitetank_rotate_10','totalwhitetank_rotate_11','totalwhitetank_rotate_12','totalwhitetank_rotate_14','totalwhitetank_rotate_20',
# 'tram_rotate_1','tram_rotate_8','tram_rotate_9','tram_rotate_19','tram_rotate_29','tram_rotate_30','tram_rotate_32',
# 'twocilinder_rotate_1','twocilinder_rotate_2','verysmallboat_rotate_6','verysmallboat_rotate_7','verysmallboat_rotate_8',
# 'vikingssailboat_rotate_1','vikingssailboat_rotate_11','vikingssailboat_rotate_12','vikingssailboat_rotate_13','vikingssailboat_rotate_17','vikingssailboat_rotate_18',
# 'vikingssailboat_rotate_20','vikingssailboat_rotate_21','vikingssailboat_rotate_22','vikingssailboat_rotate_25','vikingssailboat_rotate_26',
# 'vikingssailboat_rotate_33','vikingssailboat_rotate_36','vikingssailboat_rotate_37','vikingssailboat_rotate_38','whitebigdoor_rotate_1',
# 'white_rope_rotate_1','whiteandredlongship_rotate_1','whiteandredlongship_rotate_3','whiteandredlongship_rotate_6',
# 'whitebigsailboat_rotate_1','whitebigsailboat_rotate_2','whitebigsailboat_rotate_3','whitebigsailboat_rotate_4','whitebigsailboat_rotate_5',
# 'whitebigsailboat_rotate_6','whitebigsailboat_rotate_7','whitebigsailboat_rotate_8','whitebigsailboat_rotate_9',
# 'whitebigshoes_rotate_12','whitechess_rotate_1','whitechess_rotate_2','whitechess_rotate_3','whitereceiver_rotate_5','whitereceiver_rotate_6'
# 'whitecontainer_rotate_1','whitecontainer_rotate_4','whitecontainer_rotate_7','whitecontainer_rotate_2','whitereceiver_rotate_4','whitereceiver_rotate_7'
# 'whiteflagsignace_rotate_1','whitelocomotive_rotate_1','whitelocomotive_rotate_2','whitelocomotive_rotate_3','whitereceiver_rotate_3',
# 'whiteonewingsboat_rotate_5','whiteonewingsboat_rotate_6','whiteonewingsboat_rotate_7','whitereceiver_rotate_1','whitereceiver_rotate_2',
# 'whiteresearchship_rotate_2','whiteresearchship_rotate_3','whiteresearchship_rotate_4','whiteresearchship_rotate_5','whiteresearchship_rotate_6',
# 'whiteresearchship_rotate_7','whiteswing_rotate_1','whiteswing_rotate_8','whitetank2_rotate_5','whiteubot_rotate_2','whiteubot_rotate_1',
# 'wilson_100_tennis_ball_rotate_1','wilson_golf_ball_rotate_1','windex_rotate_1','windex_rotate_18','xwing_rotate_1',
# 'yellow_plastic_chain_rotate_1','yellowandgreenlocomotive_rotate_1','yellowbook_rotate_2','yellowcar_rotate_8','yellowcar_rotate_9',
# 'yellowformula1_rotate_1','yellowformula1_rotate_2','yellowformula1_rotate_4','yellowformula1_rotate_10','yellowlocomotive_rotate_27',
# 'yellowlocomotive_rotate_28','yellowlocomotive_rotate_29','yellowromanhat_rotate_4','yellowromanhat_rotate_5','yellowromanhat_rotate_6',
# 'yellowtank_rotate_1','yellowtank_rotate_4','yellowtram_rotate_4','yellowtram_rotate_5','yellowumbrella_rotate_1','yellowumbrella_rotate_11',
# 'yellowumbrella_rotate_12','yellowumbrella_rotate_15',]





class TesterGrab(GLRealtimeProgram):
    def __init__(self, poses, world,p_T_h,R,T, PoseDanyDiff,module):
        GLRealtimeProgram.__init__(self, "FilteredMVBBTEsterVisualizer")
        self.world = world
        self.poses = poses
        self.p_T_h = p_T_h
        self.h_T_p = np.linalg.inv(self.p_T_h)
        self.hand = None
        self.is_simulating = False
        self.curr_pose = None
        self.R = R
        self.t =T
        self.robot = self.world.robot(0)
        self.q_0 = self.robot.getConfig()
        self.w_T_o = None
        self.obj = None
        self.t_0 = None
        self.t0dany = None
        self.object_com_z_0 = None
        self.object_fell = None
        self.sim = None
        self.module = module
        self.running = True
        self.HandClose = False
        self.db = MVBBLoader(suffix='BinvoxVariation2')
        # self.logFile = DanyLog(suffix='logFile')
        self.kindness = None
        self.f1_contact = []
        self.f2_contact = []
        self.f3_contact = []
        self.crashing_states = []
        self.PoseDany = PoseDanyDiff
        self.danyK = []

        try:
            state = open('state.dump','r')
            self.crashing_states = pickle.load(state)
        except:
            pass

    def display(self):
        """ Draw a desired pose and the end-effector pose """
        if self.running:
            self.world.drawGL()

            for i in range(len(self.poses)):
                for pose in self.poses:
                    draw_GL_frame(pose, color=(0.5,0.5,0.5))
                if self.curr_pose is not None:
                    T = se3.from_homogeneous(self.curr_pose)
                    draw_GL_frame(T)

                hand_xform = get_moving_base_xform(self.robot)
                w_T_p_np = np.array(se3.homogeneous(hand_xform)).dot(self.h_T_p)
                w_T_p = se3.from_homogeneous(w_T_p_np)
                draw_GL_frame(w_T_p)

    def idle(self):
        if not self.running:
            return

        if self.world.numRigidObjects() > 0:
            self.obj = self.world.rigidObject(0)
        elif self.obj is None:
            return

        if not self.is_simulating:
            if len(self.poses) > 0:
                # self.curr_pose = np.array(se3.homogeneous(self.poses.pop(0)))
                self.curr_pose = self.poses.pop(0)
                # vis.show(hidden=False)
                # print "Simulating Next Pose Grasp"
                # print self.curr_pose
            else:
                # print "Done testing all", len(self.poses+self.poses_variations), "poses for object", self.obj.getName()
                # print "Quitting"
                self.running = False
                vis.show(hidden=True)
                return

            self.obj.setTransform(self.R, [0,0,0])
            self.obj.setVelocity([0., 0., 0.],[0., 0., 0.])
            # self.obj.setVelocity([0,0,0,e0])
            self.w_T_o = np.array(se3.homogeneous(self.obj.getTransform()))
            # embed()
            pose_se3 = se3.from_homogeneous(self.w_T_o.dot(self.curr_pose).dot(self.p_T_h))
            self.robot.setConfig(self.q_0)
            set_moving_base_xform(self.robot, pose_se3[0], pose_se3[1])

            if self.sim is None:
                self.sim = SimpleSimulator(self.world)
                self.sim.enableContactFeedbackAll()
                ##uncomment to see the log file
                # n = len(self.poses)+len(self.poses_variations) - len(self.all_poses)
                # self.sim.log_state_fn="simulation_state_" + self.obj.getName() + "_%03d"%n + ".csv"
                # self.sim.log_contact_fn="simulation_contact_"+ self.obj.getName() + "_%03d"%n + ".csv"
                # self.sim.beginLogging()

                self.hand = self.module.HandEmulator(self.sim,0,6,6)
                self.sim.addEmulator(0, self.hand)
                # self.obj.setVelocity([0., 0., 0.],[0., 0., 0.])
                # the next line latches the current configuration in the PID controller...
                self.sim.controller(0).setPIDCommand(self.robot.getConfig(), self.robot.getVelocity())

                # setup the preshrink
                visPreshrink = False  # turn this to true if you want to see the "shrunken" models used for collision detection
                for l in range(self.robot.numLinks()):
                    self.sim.body(self.robot.link(l)).setCollisionPreshrink(visPreshrink)
                for l in range(self.world.numRigidObjects()):
                    self.sim.body(self.world.rigidObject(l)).setCollisionPreshrink(visPreshrink)

            self.object_com_z_0 = getObjectGlobalCom(self.obj)[2]
            self.object_fell = False
            self.t_0 = self.sim.getTime()
            self.t0dany = self.sim.getTime()
            self.is_simulating = True

        if self.is_simulating:
            t_lift = 1.3 # when to lift
            d_lift = 1.0 # duration
            object_com_z = getObjectGlobalCom(self.obj)[2]
            hand_curr_pose = get_moving_base_xform(self.robot)
            pose_se3 = se3.from_homogeneous(self.w_T_o.dot(self.curr_pose).dot(self.p_T_h))

            if self.sim.getTime() - self.t_0 == 0:
                print "Closing hand"
                # self.hand.setCommand([0.2,0.2,0.2,0]) #TODO chiudila incrementalmente e controlla le forze di contatto
                hand_close = np.array([0.1,0.1,0.1,0])
                hand_open = np.array([1.0,1.0,1.0,0])
                step_size = 0.01
                while(self.HandClose == False):
                    d = vectorops.distance(hand_open, hand_close)
                    # print"d",d
                    n_steps = int(math.ceil(d / step_size))
                    # print"n_steps",n_steps
                    if n_steps == 0:    # if arrived
                        self.hand.setCommand([0.2,0.2,0.2,0])
                        self.HandClose = True
                    for i in range(n_steps):
                        hand_temp = vectorops.interpolate(hand_open,hand_close,float(i+1)/n_steps)
                        self.hand.setCommand([hand_temp[0] ,hand_temp[1] ,hand_temp[2] ,0])
                        self.sim.simulate(0.01)
                        self.sim.updateWorld()
                        FC = get_contact_forces_and_jacobians(self.robot,self.world,self.sim)
                        n = len(self.poses)
                        # +len(self.poses_variations) - len(self.all_poses)
                        # print"pose", n, "contact forces@t:", self.sim.getTime(), "-", FC
                        if hand_temp[0] <= hand_close[0] and hand_temp[1] <= hand_close[1] and hand_temp[2] <= hand_close[2]:
                            # print"qui"
                            self.HandClose = True
                            break

            elif (self.sim.getTime() - self.t_0) >= t_lift and (self.sim.getTime() - self.t_0) <= t_lift+d_lift:
                # print "Lifting"
                pose_se3 = se3.from_homogeneous(self.w_T_o.dot(self.curr_pose).dot(self.p_T_h))
                t_i = pose_se3[1]
                t_f = vectorops.add(t_i, (0,0,0.2))
                u = np.min((self.sim.getTime() - self.t_0 - t_lift, 1))
                send_moving_base_xform_PID(self.sim.controller(0), pose_se3[0], vectorops.interpolate(t_i, t_f ,u))
                timeDany = self.sim.getTime() - self.t_0
                self.kindness = Differential(self.robot, self.obj, self.PoseDany, timeDany)
                # print self.kindness
                self.danyK.append(self.kindness)
                self.PoseDany = RelativePosition(self.robot, self.obj)

            if (self.sim.getTime() - self.t_0) >= t_lift and (self.sim.getTime() - self.t_0) >= t_lift+d_lift:# wait for a lift before checking if object fell
                d_hand = hand_curr_pose[1][2] - pose_se3[1][2]
                d_com = object_com_z - self.object_com_z_0
                if (d_hand - d_com > 0.1) and (self.kindness >= 1E-4):
                    self.object_fell = True # TODO use grasp quality evaluator from Daniela
                    print "!!!!!!!!!!!!!!!!!!"
                    print "Object fell"
                    print "!!!!!!!!!!!!!!!!!!"
                    # Draw_Grasph(self.danyK)
                    # del self.danyK
                # else:
                #     Draw_Grasph(self.danyK)
                #     del self.danyK

            self.sim.simulate(0.01)
            self.sim.updateWorld()

            if not vis.shown() or (self.sim.getTime() - self.t_0) >= 2.5 or self.object_fell:
                if vis.shown(): # simulation stopped because it was succesfull
                    # print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                    # print "Saving grasp, object fall status:", "fallen" if self.object_fell else "grasped"
                    # print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

                    self.db.save_score(self.world.rigidObject(0).getName(), self.curr_pose, not self.object_fell,self.kindness)
                    # self.logFile.save_score(self.world.rigidObject(0).getName(), self.curr_pose, not self.object_fell,self.obj.getVelocity(), self.robot.getVelocity(), self.f1_contact,self.f2_contact,self.f3_contact)
                    if len(self.crashing_states) > 0:
                        self.crashing_states.pop()
                    state = open('state.dump','w')
                    pickle.dump(self.crashing_states, state)
                    state.close()
                # vis.show(hidden=True)
                self.is_simulating = False
                self.sim = None
                self.HandClose = False

def getObjectGlobalCom(obj):
    return se3.apply(obj.getTransform(), obj.getMass().getCom())


def Read_Poses(object_list,num,vector_set):

    nome = object_list + '_rotate_' + str(num)
    obj_dataset = '3DCNN/NNSet/Pose/PoseVariation/%s.csv'%(nome)
    print obj_dataset
    
    # try:
    with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
        file_reader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
        for row in file_reader:
            T = row[9:12]
            pp = row[:9]
            vector_set.append((pp,T))


def launch_test_mvbb_filtered(robotname, object_list):
    """Launches a very simple program that spawns an object from one of the
    databases.
    It launches a visualization of the mvbb decomposition of the object, and corresponding generated poses.
    It then spawns a hand and tries all different poses to check for collision
    """

    world = WorldModel()
    world.loadElement("data/terrains/plane.env")
    robot = make_moving_base_robot(robotname, world)
    xform = resource.get("default_initial_%s.xform" % robotname, description="Initial hand transform",
                         default=se3.identity(), world=world, doedit=False)
    
    for object_name in object_list:
        obj = None
        for object_set, objects_in_set in objects.items():
            # print objects_in_set
            if object_name in objects_in_set:
                # if world.numRigidObjects() > 0:
                #     world.remove(world.rigidObject(0))
                # count the number of mesh in the folder
                directory = 'data/objects/voxelrotate/%s/%s'%(object_set,object_name)
                list_temp = os.listdir(directory) # dir is your directory path
                number_files = len(list_temp)
                # embed()
                #take the relative pose and mesh to simualte it
                if number_files > 2:
                    for i in range(1,number_files):
                        nome = object_name + '_rotate_' +str(i)
                        if nome in done :
                            continue
                        if world.numRigidObjects() > 0:
                            world.remove(world.rigidObject(0))
                        

                        
                        obj = make_objectRotate(object_set,object_name, world,i)
                        
                        poses = [] #w_T_p_ro
                        Read_Poses(object_name,i,poses)


                        if obj is None:
                            continue
                        R,t = obj.getTransform()
                        # obj.setTransform(R, [0,0,0]) #[0,0,0] or t?
                        w_T_o = np.array(se3.homogeneous((R,[0,0,0]))) # object is at origin

                        p_T_h = np.array(se3.homogeneous(xform))

                        # poses_h = []

                        # for j in range(len(poses)):
                        #     poses_h.append((w_T_o.dot(np.array(se3.homogeneous(poses[j]))).dot(p_T_h)))

                        poses_h = []

                        for j in range(len(poses)):
                            poses_h.append((np.array(se3.homogeneous(poses[j]))).dot(p_T_h))

                        # embed()
                        # print "-------Filtering poses:"
                        filtered_poses = []
                        for j in range(len(poses)):
                            if not CollisionTestPose(world, robot, obj, poses_h[j]):
                                if not CollisionCheckWordFinger(robot, poses_h[j]):
                                    o_T_p= np.dot(np.linalg.inv(w_T_o),np.array(se3.homogeneous(poses[j])))
                                    filtered_poses.append(o_T_p)

                        if len(filtered_poses) == 0:
                            print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                            print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                            print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                            print "Filtering returned 0 feasible poses"
                            print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                            print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                            print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                            continue
                        embed()
                        # create a hand emulator from the given robot name
                        module = importlib.import_module('plugins.' + robotname)
                        R,t = obj.getTransform()
                        # emulator takes the robot index (0), start link index (6), and start driver index (6)
                        PoseDanyDiff = RelativePosition(robot,obj)
                        program = TesterGrab(filtered_poses,
                                                               world,
                                                               p_T_h,
                                                               R,
                                                               t,
                                                               PoseDanyDiff,
                                                               module)
                        vis.setPlugin(None)
                        vis.setPlugin(program)
                        program.reshape(800, 600)
                        # vis.lock()
                        vis.show()
                        # vis.unlock()
                        # this code manually updates the visualization
                        t0= time.time()
                        while vis.shown():
                            # time.sleep(0.1)
                            t1 = time.time()
                            time.sleep(max(0.01-(t1-t0),0.001))
                            t0 = t1
    return

if __name__ == '__main__':
    all_objects = []
    for dataset in objects.values():
        all_objects += dataset



    try:
        objname = sys.argv[1]
        launch_test_mvbb_filtered("reflex_col", [objname])
    except:
        launch_test_mvbb_filtered("reflex_col", all_objects)