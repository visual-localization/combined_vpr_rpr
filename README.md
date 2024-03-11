# 10/03/2024:
### Organization of Data 
- Should be changed as current organization is complicated - details are TBD

### Coordinate system of translation and orientation
- The current translation system is in the coordinate system depicted below with reference to orientation. => Need to re-orientated to get absolute position

![image](https://github.com/visual-localization/combined_vpr_rpr/assets/74974626/d10303d2-0f5c-4580-80ed-cc7c65ca7df9)

### Mapfree Dataset coordinate system
- Quaternion and Translation is given in the camera world coordinate system.
- In other words, from a point in the camera coordinate system of an image, its absolute position can be calculated as:

$abs_{point} = (cam_{point} - t) * q^{-1}$

# 11/03/2024
### Official Coordinate system to save to Scene Object
- Translation should be organized into the system below, in the world coordinate system
![image](https://github.com/visual-localization/combined_vpr_rpr/assets/74974626/7dd17648-bbb0-43f3-8dd4-3b310abc9903)
- Orientation should be from camera coordinate system to world.
- To validate your results, checks if this formula is correct.
  $abs_{point} = cam_{point}*q + t$
