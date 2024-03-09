from transforms3d.quaternions import mat2quat, qinverse, qmult, rotate_vector, quat2mat

from data import Scene

def convert_pose(R,t,db_img:Scene):
    R_ref = db_img["rotation"]
    t_ref = db_img["translation"]
    
    R_diff = mat2quat(R).reshape(-1)
    t_diff = t.flatten()
    
    R_final = qmult(R_diff,R_ref)
    t_final = t_diff + rotate_vector(t_ref,R_diff)
    return R_final,t_final
    