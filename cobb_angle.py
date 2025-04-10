import numpy as np
import math

def _create_angles_dict(pt, mt, tl):
  """
  pt,mt,tl: tuple(2) that contains: (angle, [idxTop, idxBottom])
  """
  return {
    "pt": {
        "angle": pt[0],
        "idxs": [pt[1][0], pt[1][1]],
    },
    "mt": {
        "angle": mt[0],
        "idxs": [mt[1][0], mt[1][1]],
    },
    "tl": {
        "angle": tl[0],
        "idxs": [tl[1][0], tl[1][1]],
    }
  }

def _isS(p):
    num = len(p)
    ll = np.zeros([num-2,1])
    for i in range(num-2):
        ll[i] = (p[i][1]-p[num-1][1])/(p[0][1]-p[num-1][1]) - (p[i][0]-p[num-1][0])/(p[0][0]-p[num-1][0])

    flag = np.sum(np.sum(np.dot(ll,ll.T))) != np.sum(np.sum(abs(np.dot(ll,ll.T))))
    return(flag)


def cobb_angle_cal(landmark_xy, image_shape):
  """
  `landmark_xy`: number[n]. [x1,x2,...,xn,y1,y2,...,yn], where 
  - `n` is even.
  - 0 <= x <= W
  - 0 <= y <= H 
  `image_shape`: (HEIGHT, WIDTH, CHANNELS) *only HEIGHT is important

  Returns: Tuple(4): cobb_angles_list, angles_with_pos, curve_type, midpoint_lines.
  - `cobb_angles_list` - For evaluating with ground-truth: ex. [0.50, 0.11, 0.33].
  - `angles_with_pos` - dict of "pt", "mt", "tl", each with values for "angle" and "idxs".
  - `curve_type` - "S" or "C".
  - `midpoint_lines` - list of mid point line coordinates: ex. [[[x,y][x,y]], [[x,y][x,y]], ...].
  """
  landmark_xy = list(landmark_xy) # input is list
  ap_num = int(len(landmark_xy)/2) # number of points
  vnum = int(ap_num / 4) # number of verts
  
  first_half = landmark_xy[:ap_num]
  second_half = landmark_xy[ap_num:]

  # Values this function returns
  cob_angles = np.zeros(3)  
  angles_with_pos = {}
  curve_type = None

  # Midpoints (2 points per vertebra)
  mid_p_v = []
  for i in range(int(len(landmark_xy)/4)):
      x = first_half[2*i: 2*i+2]
      y = second_half[2*i: 2*i+2]
      row = [(x[0] + x[1]) / 2, (y[0] + y[1]) / 2]
      mid_p_v.append(row)

  mid_p = []
  for i in range(int(vnum)):
      x = first_half[4*i: 4*i+4]
      y = second_half[4*i: 4*i+4]
      point1 = [(x[0] + x[2]) / 2, (y[0] + y[2]) / 2]
      point2 = [(x[3] + x[1]) / 2, (y[3] + y[1]) / 2]
      mid_p.append(point1)
      mid_p.append(point2)

  # Line and Slope
  vec_m = []
  for i in range(int(len(mid_p)/2)):
    points = mid_p[2*i: 2*i+2]
    row = [points[1][0]-points[0][0], points[1][1]-points[0][1]]
    vec_m.append(row)

  mod_v = []
  for i in vec_m:
      row = [i[0]*i[0], i[1]*i[1]]
      mod_v.append(row)
  dot_v = np.dot(np.matrix(vec_m), np.matrix(vec_m).T)
  mod_v = np.sqrt(np.sum(np.matrix(mod_v), axis=1))

  dot_v = np.dot(np.matrix(vec_m), np.matrix(vec_m).T)

  slopes = []
  for i in vec_m:
      slope = i[1]/i[0]
      slopes.append(slope)

  angles = np.clip(dot_v/np.dot(mod_v, mod_v.T), -1, 1)
  angles = np.arccos(angles)

  maxt = np.amax(angles, axis = 0)
  pos1 = np.argmax(angles, axis = 0)

  pt, pos2 = np.amax(maxt), np.argmax(maxt)

  pt = pt*180/math.pi
  cob_angles[0] = pt

  if(_isS(mid_p_v)==False):
    mod_v1 = np.sqrt(np.sum(np.multiply(np.matrix(vec_m[0]), np.matrix(vec_m[0]))))
    mod_vs1 = np.sqrt(np.sum(np.multiply(np.matrix(vec_m[pos2]), np.matrix(vec_m[pos2])), axis=1))
    mod_v2 = np.sqrt(np.sum(np.multiply(np.matrix(vec_m[int(vnum-1)]), np.matrix(vec_m[int(vnum-1)])), axis=1))
    mod_vs2 = np.sqrt(np.sum(np.multiply(vec_m[pos1.item((0, pos2))], vec_m[pos1.item((0, pos2))])))
       
    dot_v1 = np.dot(np.array(vec_m[0]), np.array(vec_m[pos2]).T)
    dot_v2 = np.dot(np.array(vec_m[int(vnum-1)]), np.array(vec_m[pos1.item((0, pos2))]).T)
        
    mt = np.arccos(np.clip(dot_v1/np.dot(mod_v1, mod_vs1.T), -1, 1))
    tl = np.arccos(np.clip(dot_v2/np.dot(mod_v2, mod_vs2.T), -1, 1))  

    mt = mt*180/math.pi
    tl = tl*180/math.pi
    cob_angles[1] = mt
    cob_angles[2] = tl

    # DETECTION CASE 1: Spine Type C
    angles_with_pos = _create_angles_dict(mt=(float(pt), [pos2, pos1.A1.tolist()[pos2]]), pt=(float(mt), [0, int(pos2)]), tl=(float(tl), [pos1.A1.tolist()[pos2], vnum-1]))
    curve_type = "C"

  else:
    if(((mid_p_v[pos2*2][1])+mid_p_v[pos1.item((0, pos2))*2][1]) < image_shape[0]):
        #Calculate Upside Cobb Angle
        mod_v_p = np.sqrt(np.sum(np.multiply(vec_m[pos2], vec_m[pos2])))
        mod_v1 = np.sqrt(np.sum(np.multiply(vec_m[0:pos2], vec_m[0:pos2]), axis=1))
        dot_v1 = np.dot(np.array(vec_m[pos2]), np.array(vec_m[0:pos2]).T)

        angles1 = np.arccos(np.clip(dot_v1/np.dot(mod_v_p, mod_v1.T), -1, 1))
        CobbAn1, pos1_1 = np.amax(angles1, axis = 0), np.argmax(angles1, axis = 0)
        mt = CobbAn1*180/math.pi
        cob_angles[1] = mt

        #Calculate Downside Cobb Angle
        mod_v_p2 = np.sqrt(np.sum(np.multiply(vec_m[pos1.item((0, pos2))], vec_m[pos1.item((0, pos2))])))
        mod_v2 = np.sqrt(np.sum(np.multiply(vec_m[pos1.item((0, pos2)):int(vnum)], vec_m[pos1.item((0, pos2)):int(vnum)]), axis=1))
        dot_v2 = np.dot(np.array(vec_m[pos1.item((0, pos2))]), np.array(vec_m[pos1.item((0, pos2)):int(vnum)]).T)
        
        angles2 = np.arccos(np.clip(dot_v2/np.dot(mod_v_p2, mod_v2.T), -1, 1))
        CobbAn2, pos1_2 = np.amax(angles2, axis = 0), np.argmax(angles2, axis = 0)
        tl = CobbAn2*180/math.pi
        cob_angles[2] = tl

        pos1_2 = pos1_2 + pos1.item((0, pos2)) - 1

        # DETECTION CASE 2: Spine Type S, Up and Bottom
        # print("case 2")
        angles_with_pos = _create_angles_dict(mt=(float(pt), [pos2, pos1.A1.tolist()[pos2]]), pt=(float(mt), [int(pos1_1), int(pos2)]), tl=(float(tl), [pos1.A1.tolist()[pos2], int(pos1_2)]))
        curve_type = "S"
    
    else:
        #Calculate Upside Cobb Angle
        mod_v_p = np.sqrt(np.sum(np.multiply(vec_m[pos2], vec_m[pos2])))
        mod_v1 = np.sqrt(np.sum(np.multiply(vec_m[0:pos2], vec_m[0:pos2]), axis=1))
        dot_v1 = np.dot(np.array(vec_m[pos2]), np.array(vec_m[0:pos2]).T)

        angles1 = np.arccos(np.clip(dot_v1/np.dot(mod_v_p, mod_v1.T), -1, 1))
        CobbAn1 = np.amax(angles1, axis = 0)
        pos1_1 = np.argmax(angles1, axis = 0)
        mt = CobbAn1*180/math.pi
        cob_angles[1] = mt

        #Calculate Upper Upside Cobb Angle
        mod_v_p2 = np.sqrt(np.sum(np.multiply(vec_m[pos1_1], vec_m[pos1_1])))
        mod_v2 = np.sqrt(np.sum(np.multiply(vec_m[0:pos1_1+1], vec_m[0:pos1_1+1]), axis=1))
        dot_v2 = np.dot(np.array(vec_m[pos1_1]), np.array(vec_m[0:pos1_1+1]).T)
        
        angles2 = np.arccos(np.clip(dot_v2/np.dot(mod_v_p2, mod_v2.T), -1, 1))
        CobbAn2, pos1_2 = np.amax(angles2, axis = 0), np.argmax(angles2, axis = 0)
        tl = CobbAn2*180/math.pi
        cob_angles[2] = tl
        # pos1_2 = pos1_2 + pos1.item((0, pos2)) - 1
        
        # DETECTION CASE 3: Spine Type S, Up and Bottom
        # print("case 3")
        angles_with_pos = _create_angles_dict(tl=(float(pt), [pos2, pos1.A1.tolist()[pos2]]), mt=(float(mt), [pos1_1, pos2]), pt=(float(tl), [int(pos1_2), int(pos1_1)]))
        curve_type = "S"
  
  midpoint_lines = []
  for i in range(0,int(len(mid_p)/2)):
    midpoint_lines.append([list(map(int, mid_p[i*2])), list(map(int, mid_p[i*2+1]))])

  # Remove Numpy Values
  cobb_angles_list = [float(c) for c in cob_angles]
  for key in angles_with_pos.keys():
    angles_with_pos[key]['angle'] = float(angles_with_pos[key]['angle'])
    for i in range(len(angles_with_pos[key]['idxs'])):
        angles_with_pos[key]['idxs'][i] = int(angles_with_pos[key]['idxs'][i])



  print("Cobb Angles: ", cobb_angles_list)
  print("Angles with Pos: ", angles_with_pos)
  print("Curve Type: ", curve_type)
  print("Midpoint Lines: ", midpoint_lines)
  return cobb_angles_list, angles_with_pos, curve_type, midpoint_lines












'''
Bu kod, bir X-ray görüntüsünde omurganın eğriliğini belirlemek için "Cobb açısı" olarak bilinen açıyı hesaplar. Cobb açısı, skolyoz teşhisinde kullanılan önemli bir ölçüttür. Kod, omurga eğriliğini analiz etmek için belirli noktalar arasındaki açıları hesaplar ve sonuç olarak Cobb açısını verir. İşte bu kodun nasıl çalıştığını ve ne döndürdüğünü Türkçe olarak açıklamaya çalışacağım:

Giriş Parametreleri:
landmark_xy: Noktaların (işaretçilerin) x ve y koordinatlarını içeren bir liste. Bu liste, sırasıyla ilk yarıda x koordinatlarını ve ikinci yarıda y koordinatlarını içerir.
image_shape: Görüntünün yüksekliği, genişliği ve kanallarını içeren bir tuple. Burada sadece yüksekliği (HEIGHT) önemlidir.
Döndürülen Değerler:
cobb_angles_list: Cobb açılarını içeren bir liste. Örnek: [0.50, 0.11, 0.33].
angles_with_pos: Her biri "pt", "mt", "tl" olarak adlandırılan açılar ve bu açıların hesaplandığı noktaların indekslerini içeren bir sözlük.
curve_type: Eğriliğin tipini belirtir ("S" veya "C").
midpoint_lines: Orta noktaların koordinatlarını içeren bir liste.
Hesaplama Adımları:
Nokta Sayısının ve Vertebra Sayısının Hesaplanması:

ap_num: Nokta sayısı (landmark_xy'nin uzunluğunun yarısı).
vnum: Vertebra (omur) sayısı (ap_num'un dörtte biri).
Orta Noktaların Hesaplanması:

mid_p_v ve mid_p: Omurların ve vertebraların orta noktalarını hesaplar.
Her bir vertebra için iki orta nokta hesaplanır ve bu orta noktalar kullanılarak omurga eğriliği analiz edilir.
Eğim ve Açı Hesaplamaları:

vec_m: Orta noktalar arasındaki vektörler.
mod_v ve dot_v: Bu vektörlerin büyüklükleri ve noktaların çarpımları.
angles: Vektörler arasındaki açılar (radyan cinsinden).
Maksimum Açıların Belirlenmesi:

pt, mt, ve tl: En büyük açılar ve bunların konumları (indeksleri).
cob_angles: Hesaplanan Cobb açılarını içerir.
Eğrilik Tipinin Belirlenmesi ve Sonuçların Döndürülmesi:

_isS(mid_p_v): Eğriliğin "S" mi yoksa "C" mi olduğunu belirler.
Eğrilik tipi belirlendikten sonra açılar ve indeksler angles_with_pos sözlüğüne eklenir.



Örnek Çıktılar:
Cobb Angles: [50.0, 11.0, 33.0]
Angles with Pos: {
  'pt': {'angle': 50.0, 'idxs': [0, 2]},
  'mt': {'angle': 11.0, 'idxs': [1, 3]},
  'tl': {'angle': 33.0, 'idxs': [2, 4]}
}
Curve Type: S
Midpoint Lines: [[[x1,y1], [x2,y2]], [[x3,y3], [x4,y4]], ...]
Açıklamalar:
Cobb Angles: Üç farklı Cobb açısını içerir.
Angles with Pos: Her bir açının değerini ve bu açının hesaplandığı vertebra indekslerini içerir.
Curve Type: Eğriliğin tipini belirler ("S" veya "C").
Midpoint Lines: Orta noktaların koordinatlarını içerir.
Kodun genel işleyişi bu şekildedir. Omurga eğriliğini belirlemek için vertebralar arasındaki açılar hesaplanır ve bu açılar kullanılarak Cobb açıları ve eğrilik tipi belirlenir.

'''



