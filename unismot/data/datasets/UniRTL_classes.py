#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# Copyright (c) Lian Zhang and its affiliates.

UniRTL_CLASSES = (
    "person",
    "car",
    "bicycle",
    "electromobile",
    "tricycle",
    "football",
    "bus",
    "truck",
    "helmet",
    "goose",
    "tennisracket",
    "taxi",
    "license",
    "mouse",
    "mask",
    "cap",
    "minibus",
    "skeet",
    "plasticfilm",
    "bag",
    "motorcycle",
    "dog",
    "duck",
    "mitt",
    "baseball_bat",
    "baseball_hat",
    "bird",
    "carton",
    "shoe",
    "balance_car",
    "cat",
    "handbag",
    "minicar",
    "kite",
    "umbrella",
)

GTOT_CLASSES=('BlackCar', 'BlackSwan1', 'BlueCar', 'BusScale', 'BusScale1', 'Crossing', 'Cycling', 'DarkNig', 'Exposure2', 'Exposure4', 'FastCarNig', 'FastMotor', 'FastMotorNig', 'Football', 'GarageHover', 'Gathering', 'GoTogether', 'Jogging', 'LightOcc', 'Minibus', 'Minibus1', 'MinibusNig', 'MinibusNigOcc', 'MotorNig', 'Motorbike', 'Motorbike1', 'OccCar-1', 'OccCar-2', 'Otcbvs', 'Otcbvs1', 'Pool', 'Quarreling', 'RainyCar1', 'RainyCar2', 'RainyMotor1', 'RainyMotor2', 'RainyPeople', 'Running', 'Torabi', 'Torabi1', 'Tricycle', 'Walking', 'WalkingNig', 'WalkingNig1', 'WalkingOcc', 'carNig', 'crowdNig', 'fastCar2', 'occBike', 'tunnel')
RGBT234_CLASSES=('afterrain', 'aftertree', 'baby', 'baginhand', 'baketballwaliking', 'balancebike', 'basketball2', 'bicyclecity', 'bike', 'bikeman', 'bikemove1', 'biketwo', 'blackwoman', 'blueCar', 'bluebike', 'boundaryandfast', 'bus6', 'call', 'car', 'car10', 'car20', 'car3', 'car37', 'car4', 'car41', 'car66', 'carLight', 'caraftertree', 'carnotfar', 'carnotmove', 'carred', 'child', 'child1', 'child3', 'child4', 'children2', 'children3', 'children4', 'crossroad', 'crouch', 'cycle1', 'cycle2', 'cycle3', 'cycle4', 'cycle5', 'diamond', 'dog', 'dog1', 'dog10', 'dog11', 'elecbike', 'elecbike10', 'elecbike2', 'elecbike3', 'elecbikechange2', 'elecbikeinfrontcar', 'elecbikewithhat', 'elecbikewithlight', 'elecbikewithlight1', 'face1', 'floor-1', 'flower1', 'flower2', 'fog', 'fog6', 'glass', 'glass2', 'graycar2', 'green', 'greentruck', 'greyman', 'greywoman', 'guidepost', 'hotglass', 'hotkettle', 'inglassandmobile', 'jump', 'kettle', 'kite2', 'kite4', 'luggage', 'man2', 'man22', 'man23', 'man24', 'man26', 'man28', 'man29', 'man3', 'man4', 'man45', 'man5', 'man55', 'man68', 'man69', 'man7', 'man8', 'man88', 'man9', 'manafterrain', 'mancross', 'mancross1', 'mancrossandup', 'mandrivecar', 'manfaraway', 'maninblack', 'maninglass', 'maningreen2', 'maninred', 'manlight', 'manoccpart', 'manonboundary', 'manonelecbike', 'manontricycle', 'manout2', 'manup', 'manwithbag', 'manwithbag4', 'manwithbasketball', 'manwithluggage', 'manwithumbrella', 'manypeople', 'manypeople1', 'manypeople2', 'mobile', 'night2', 'nightcar', 'nightrun', 'nightthreepeople', 'notmove', 'oldman', 'oldman2', 'oldwoman', 'orangeman1', 'people', 'people1', 'people3', 'playsoccer', 'push', 'rainingwaliking', 'raningcar', 'redbag', 'redcar', 'redcar2', 'redmanchange', 'rmo', 'run', 'run1', 'run2', 'scooter', 'shake', 'shoeslight', 'single1', 'single3', 'soccer', 'soccer2', 'soccerinhand', 'straw', 'stroller', 'supbus', 'supbus2', 'takeout', 'tallman', 'threeman', 'threeman2', 'threepeople', 'threewoman2', 'together', 'toy1', 'toy3', 'toy4', 'tree2', 'tree3', 'tree5', 'trees', 'tricycle', 'tricycle1', 'tricycle2', 'tricycle6', 'tricycle9', 'tricyclefaraway', 'tricycletwo', 'twoelecbike', 'twoelecbike1', 'twoman', 'twoman1', 'twoman2', 'twoperson', 'twowoman', 'twowoman1', 'walking40', 'walking41', 'walkingman', 'walkingman1', 'walkingman12', 'walkingman20', 'walkingman41', 'walkingmantiny', 'walkingnight', 'walkingtogether', 'walkingtogether1', 'walkingtogetherright', 'walkingwithbag1', 'walkingwithbag2', 'walkingwoman', 'whitebag', 'whitecar', 'whitecar3', 'whitecar4', 'whitecarafterrain', 'whiteman1', 'whitesuv', 'woamn46', 'woamnwithbike', 'woman', 'woman1', 'woman100', 'woman2', 'woman3', 'woman4', 'woman48', 'woman6', 'woman89', 'woman96', 'woman99', 'womancross', 'womanfaraway', 'womaninblackwithbike', 'womanleft', 'womanpink', 'womanred', 'womanrun', 'womanwithbag6', 'yellowcar')
