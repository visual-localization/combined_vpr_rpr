MAPFREE_RESIZE = (540,720)
MIXVPR_RESIZE = (320,320)
CAM_RESIZE = (852,480)
GSV_RESIZE = (400,300)
EMAT_RANSAC={
    "pix_threshold":2.0,
    "confidence":0.9999,
    "scale_threshold":0.1 
}
PNP = {
    "ransac_iter":1000,
    "reprojection_inlier_threshold":3,
    "confidence":0.9999
}
FRUSTUM_THRESHOLD = 1.95
ANGLE_THRESHOLD = 80
CAM_LANDMARK_SCENE_BUNDLE = [
    "GreatCourt",
    "KingsCollege",
    "OldHospital",
    "ShopFacade",
    "StMarysChurch",
    "Street"
]