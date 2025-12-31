import json
import random
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add, brepbndlib
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse,BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Section
from OCC.Core.Message import Message_ProgressRange
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.gp import gp_Trsf, gp_Pnt, gp_Ax1, gp_Vec, gp_Dir
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Quantity import Quantity_NOC_RED, Quantity_NOC_GREEN, Quantity_NOC_BLUE, Quantity_NOC_YELLOW ,Quantity_Color
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SHAPE
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, BRepGProp_Face, brepgprop
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeWire
from OCC.Core.BRepFill import BRepFill_Filling
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Edge, topods_Edge, topods_Vertex, topods

import createdata.fix_index as fix_index

# 读取STP文件
def read_stp(file_path):
    reader = STEPControl_Reader()
    reader.ReadFile(file_path)
    status = reader.TransferRoots()
    if status == IFSelect_RetDone:
        return reader.Shape()
    else:
        raise Exception("Failed to read STEP file")
    
def write_to_step(shape, file_path):
    # 创建STEP文件写入器
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(file_path)
    if status == IFSelect_RetDone:
        print(f"模型已成功写入到{file_path}")
    else:
        print(f"写入模型到{file_path}时出错")


def fill_holes_in_face(face):
    # 创建一个缝合工具
    sewing = BRepBuilderAPI_Sewing(0.01)  # 容差值可以根据需要调整

    # 获取面的边界
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    edges = []
    while explorer.More():
        edge = topods_Edge(explorer.Current())
        edges.append(edge)
        explorer.Next()

    # 创建一个新的面
    new_face = BRepBuilderAPI_MakeFace(face).Face()

    # 填充孔洞
    for edge in edges:
        filling = BRepFill_Filling()
        filling.Add(edge)
        filling.Build()
        if filling.IsDone():
            new_wire = BRepBuilderAPI_MakeWire(filling.Wire()).Wire()
            sewing.Add(new_wire)

    # 执行缝合
    sewing.Perform()
    sewn_shell = sewing.SewedShape()

    # 将缝合后的壳转换为面
    sewn_face = BRepBuilderAPI_MakeFace(sewn_shell).Face()

    return sewn_face
def vertices_are_close(v1, v2, tolerance=1e-7):
    """检查两个顶点是否足够接近，认为是相同的点"""
    p1 = BRep_Tool.Pnt(topods.Vertex(v1))
    p2 = BRep_Tool.Pnt(topods.Vertex(v2))
    return p1.IsEqual(p2, tolerance)

def find_next_connected_edge(edges, current_edge, used_edges):
    """查找与当前边相连但尚未使用的下一条边"""
    # 获取当前边的端点
    explorer = TopExp_Explorer(current_edge, TopAbs_VERTEX)
    start_vertex = explorer.Current()
    explorer.Next()
    end_vertex = explorer.Current()

    for edge in edges:
        if edge not in used_edges and edge != current_edge:
            explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
            edge_start_vertex = explorer.Current()
            explorer.Next()
            edge_end_vertex = explorer.Current()

            # 检查当前边的终点是否与候选边的起点或终点相同
            if vertices_are_close(end_vertex, edge_start_vertex, 1e-7) or \
               vertices_are_close(end_vertex, edge_end_vertex, 1e-7):
                return edge

    return None

def get_wires_from_edges(edges):
    """从边列表中获取所有封闭的线框"""
    wires = []
    used_edges = set()
    
    while edges:  # 假设edges是可变的，且可以从中移除已使用的边
        # 选择第一条未使用的边作为起始边
        start_edge = next((edge for edge in edges if edge not in used_edges), None)
        if start_edge is None:
            break
        
        mk_wire = BRepBuilderAPI_MakeWire()
        current_edge = start_edge
        used_edges.add(current_edge)
        mk_wire.Add(current_edge)
        
        while True:
            next_edge = find_next_connected_edge(edges, current_edge, used_edges)
            if next_edge is None:
                break
            
            used_edges.add(next_edge)
            mk_wire.Add(next_edge)
            current_edge = next_edge
            
        if mk_wire.IsDone():
            wire_shape = mk_wire.Wire()
            wires.append(wire_shape)
            
        # 清理已使用的边
        edges = [edge for edge in edges if edge not in used_edges]

    return wires

def calculate_area(wire):
    """计算由线框围成的面的面积"""
    face = BRepBuilderAPI_MakeFace(wire).Face()
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return props.Mass()  # 面积在这里被表示为质量

def random_planar_transform_on_contact_face(shape, base, difference, rotate=False, scale=False):
  
    explorer = TopExp_Explorer(base, TopAbs_FACE)

    max_area = -1
    max_face = None
    face_count = 0

    
    while explorer.More():
        face = explorer.Current()
        
        # 计算当前面的面积
        properties = GProp_GProps()
        brepgprop.SurfaceProperties(face, properties)
        area = properties.Mass()
        
        # 更新最大面积和对应的面
        if area > max_area:
            max_area = area
            max_face = face
            max_face_id = face_count
        
        # 移动到下一个面
        explorer.Next()
        face_count += 1
    
    # max_area = -1
    # max_face = None
    # face_count = 0
    # while explorer.More():
    #     face = explorer.Current()
    #     face_count += 1

    #     # 创建GProp_Face对象来计算面的法向量
    #     analysis_face = BRepGProp_Face(face)

    #     if face_count == 3 :
    #         max_face=face
    #         break
    #     # 移动到下一个形状
    #     explorer.Next()

    # 获取面的几何表面
    surf = BRep_Tool.Surface(max_face)
    # 创建GProp_Face对象来计算面的法向量
    analysis_face = BRepGProp_Face(max_face)
    # 获取面的参数范围
    umin, umax, vmin, vmax = analysis_face.Bounds()

    # 填充面中的孔洞
    #filled_face = fill_holes_in_face(max_face)
   # write_to_step(max_face, rf"C:\CAD识别数据增强\code\仪表盘\face.step")



    # 随机选择面上的一个点
    randU = random.uniform(umin, umax)
    randV = random.uniform(vmin, vmax)
    
    # 获取随机点的法向量和位置
    mid_point = gp_Pnt()
    norm = gp_Vec()
    analysis_face.Normal(randU, randV, mid_point, norm)
    
    #  # 获取曲面的第一阶偏导数，即切线向量---原代码
    # d1p = gp_Pnt()  # 忽略这个参数
    # du = gp_Vec()
    # dv = gp_Vec()
    # surf.D1(randU, randV, d1p, du, dv)  # 提供所有需要的参数
    
    # # 选择一个切线方向向量，并构造一个与法向量垂直的随机方向向量
    # tangent_vec = du.Crossed(gp_Vec(1., 0., 0.)) if abs(norm.Dot(gp_Vec(1., 0., 0.))) < 1 else du.Crossed(gp_Vec(0., 1., 0.))
    # tangent_vec.Normalize()
    # 生成一个与法向量垂直的随机方向向量
    def get_tangent_vector(norm):
        # 选择一个与法向量不平行的向量
        if abs(norm.Dot(gp_Vec(1., 0., 0.))) < 1e-6:
            return norm.Crossed(gp_Vec(1., 0., 0.))
        elif abs(norm.Dot(gp_Vec(0., 1., 0.))) < 1e-6:
            return norm.Crossed(gp_Vec(0., 1., 0.))
        else:
            return norm.Crossed(gp_Vec(0., 0., 1.))
    
    tangent_vec1 = get_tangent_vector(norm)
    
    if tangent_vec1.Magnitude() < 1e-6:
        raise ValueError("Failed to find a valid tangent vector.")
    
    tangent_vec1.Normalize()
    
    # 生成另一个与法向量和第一个切线向量都垂直的向量
    tangent_vec2 = norm.Crossed(tangent_vec1)
    tangent_vec2.Normalize()
    
    # 选择一个随机方向向量
    random_tangent_vec = tangent_vec1 * random.uniform(-1, 1) + tangent_vec2 * random.uniform(-1, 1)
    random_tangent_vec.Normalize()
    
    trsf_vec = random_tangent_vec.Scaled(random.uniform(-1, 1))

    bounding_box = Bnd_Box()
    brepbndlib.Add(base, bounding_box)
    xmin, ymin, zmin, xmax, ymax, zmax = bounding_box.Get()
    
    bounding_box2 = Bnd_Box()
    brepbndlib.Add(shape, bounding_box2)
    xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = bounding_box2.Get()

    tmin = min(xmin - xmin2, xmax - xmax2)
    tmax = max(xmax - xmin2, xmin - xmax2)
    ok = 0
    
    transformed_shape=None
    # 尝试平移直到两个模型相交
    while True:
        t = random.uniform(tmin, tmax)
        
        trsf = gp_Trsf()
        trsf.SetTranslationPart(trsf_vec.Scaled(t))
        
        transformed_shape = BRepBuilderAPI_Transform(shape, trsf).Shape()
        
        # 检查是否相交
        intersection1 = BRepAlgoAPI_Section(transformed_shape, base).Shape()
        intersection2 = BRepAlgoAPI_Section(transformed_shape, difference).Shape()
        explorer1 = TopExp_Explorer(intersection1, TopAbs_VERTEX)
        explorer2 = TopExp_Explorer(intersection2, TopAbs_VERTEX)
        if explorer1.More() and not explorer2.More():
            break
        elif ok > 30:
            return shape    
        else:
            ok += 1
            continue 

    intersection = BRepAlgoAPI_Section(transformed_shape, base).Shape()
    
    if intersection.IsNull():
        rotate=False
        scale=False
        raise ValueError("No intersection found between shape and base.")
    if rotate or scale:
        # 找到交集中面积最大的面

        explorer = TopExp_Explorer(intersection, TopAbs_EDGE)
        edges = []
        while explorer.More():
            edge = explorer.Current()
            edges.append(edge)
            explorer.Next()
        # 获取所有封闭的线框
        wires = get_wires_from_edges(edges)

        # 计算每个线框的面积并找出最大面积的线框
        max_area = 0
        outer_wire = None
        for wire in wires:
            area = calculate_area(wire)
            if area > max_area:
                max_area = area
                outer_wire = wire
        outer_face=None
        # 使用最大面积的线框创建面
        if outer_wire is not None:
            outer_face = BRepBuilderAPI_MakeFace(outer_wire).Face()
        else:
            # print("没有找到合适的外轮廓")
            # return transformed_shape
            outer_face = BRepBuilderAPI_MakeFace(outer_wire).Face()
        # 确保interFace是TopoDS_Face类型
        #if not isinstance(interFace, TopoDS_Face):
            #write_to_step(interFace, rf"C:\CAD识别数据增强\code\中面\data\step\k.step")
            #raise TypeError("interFace must be of type TopoDS_Face")
        # 获取面的几何表面和面的法向量
        analysis_face = BRepGProp_Face(outer_face)
        umin, umax, vmin, vmax = analysis_face.Bounds()

        # 计算面的质心
        properties = GProp_GProps()
        brepgprop_SurfaceProperties(outer_face, properties)
        centroid = properties.CentreOfMass()

        # 随机放缩倍率
        scale_factor = random.uniform(0.75, 1.5)
            # 随机选择面上的一个点
        randU = random.uniform(umin, umax)
        randV = random.uniform(vmin, vmax)
        
        # 获取随机点的法向量和位置
        mid_point = gp_Pnt()
        normal_vec = gp_Vec()
        # 随机旋转角度和轴
        rotation_angle = random.uniform(-5, 5)
        analysis_face.Normal(randU, randV, mid_point, normal_vec)
        # 将 gp_Vec 转换为 gp_Dir
        normal_dir = gp_Dir(normal_vec)
        rotation_axis = gp_Ax1(centroid, normal_dir)

        # 创建变换对象
        trsf = gp_Trsf()
        if scale:
        # 放缩
            trsf.SetScale(centroid, scale_factor)
        if rotate:
        # 旋转
            trsf.SetRotation(rotation_axis, rotation_angle)

        # 应用变换
        transformed_shape = BRepBuilderAPI_Transform(transformed_shape, trsf).Shape()
    return transformed_shape

    
def get_surface_index(shape, face):
    # 获取表面在形状中的索引
    index = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        f = explorer.Current()
        if face.IsEqual(f):
            return index
        index += 1
        explorer.Next()
    return None

# def find_original_shape(face, shapes):
#     # 确定该表面属于哪个原始模型
#     for original_shape in shapes:
#         # 遍历原始模型中的每个表面
#         explorer = TopExp_Explorer(original_shape, TopAbs_FACE)
#         while explorer.More():
#             f = explorer.Current()
#             if face.IsEqual(f):
#                 return original_shape
#             explorer.Next()
#     return None
def find_original_shape(face, shapes):
    """
    确定给定的表面属于哪个原始模型，并返回该模型及对应面的索引。

    :param face: 需要查找对应关系的面
    :param shapes: 原始模型列表
    :return: 一个元组 (original_shape, original_face_index)，如果未找到则返回 (None, None)
    """
    for original_shape in shapes:
        # 初始化面索引
        face_index = 0
        explorer = TopExp_Explorer(original_shape, TopAbs_FACE)
        while explorer.More():
            f = topods.Face(explorer.Current())
            if face.IsEqual(f):
                return original_shape, face_index  # 返回找到的原始模型和面索引
            explorer.Next()
            face_index += 1  # 更新面索引
    return None, None  # 如果没有找到匹配的面，则返回 None, None
def getNum(y,x):
    if(y==3):
        return x
    else:
        return y


def color_faces(shapes, base, mapping, x):
    colors = {
        shapes[0]: Quantity_NOC_RED,
        shapes[1]: Quantity_NOC_GREEN,
        shapes[2]: Quantity_NOC_BLUE,
        shapes[3]: Quantity_NOC_YELLOW
    }

    explorer_e = TopExp_Explorer(base, TopAbs_FACE)
    while explorer_e.More():
        face_e = topods.Face(explorer_e.Current())

        # 使用更新后的 find_original_shape 函数
        original_shape, original_face_index = find_original_shape(face_e, shapes)
        if original_shape:
            surface_index = get_surface_index(base, face_e)
            if surface_index is not None:
                original_shape_index = shapes.index(original_shape)

                # 更新 mapping，现在包括原始模型中的具体面对应索引
                mapping[str(surface_index)] = {
                    'original_shape_index': original_shape_index,
                    'original_face_index': original_face_index,
                    'value': getNum(original_shape_index, x)  # 根据需要计算的值
                }
        explorer_e.Next()

    # 获取最大的索引值
    max_index_str = max(mapping.keys(), default='0')
    max_index = int(max_index_str)

    # 从最大索引到0遍历，并填充缺失的键
    for i in range(max_index + 1):
        index_str = str(i)
        if index_str not in mapping:
            mapping[index_str] = {'original_shape_index': None, 'original_face_index': None, 'value': x}

def check_intersection(shape1, shape2):
    # 使用布尔运算检查相交
    common = BRepAlgoAPI_Section(shape1, shape2)
    common.Build()
    # 如果布尔运算的结果为空，则说明没有交集,返回true
    return common.Shape().IsNull()


def main():
    # 读取模型
    LL = 0
    # 是否有其他底座和零件
    if LL == 0:
        base = read_stp(r"C:\Users\20268\Desktop\Project\ProcessData\Data\CBF\原型\底座1.STEP")
        part_a = read_stp(r"C:\Users\20268\Desktop\Project\ProcessData\Data\CBF\原型\螺钉1.STEP")
        part_b = read_stp(r"C:\Users\20268\Desktop\Project\ProcessData\Data\CBF\原型\内六角1.STEP")
        part_c = read_stp(r"C:\Users\20268\Desktop\Project\ProcessData\Data\CBF\原型\凸台1.STEP")
    mode = 1  # 是否复制
    dizuo = 3
    progress_range = Message_ProgressRange()

    for i in range(18392,21392):
        if i >= 8000:
            mode = 2

        try:
            transformed_a = random_planar_transform_on_contact_face(part_a, base, part_a)
            transformed_b = random_planar_transform_on_contact_face(part_b, base, transformed_a)
            difference = BRepAlgoAPI_Fuse(transformed_b, transformed_a).Shape()
        except TypeError as e:
            print(f"Fuse operation failed between transformed_b and transformed_a: {e}")
            continue  # 跳过当前循环的其余部分

        try:
            transformed_c = random_planar_transform_on_contact_face(part_c, base, difference)
        except TypeError as e:
            print(f"Transform operation failed for part_c: {e}")
            continue

        try:
            intersection_ab = BRepAlgoAPI_Common(transformed_a, base, progress_range).Shape()
        except TypeError as e:
            print(f"Common operation failed between transformed_a and base: {e}")
            continue

        try:
            intersection_bc = BRepAlgoAPI_Common(transformed_b, base, progress_range).Shape()
        except TypeError as e:
            print(f"Common operation failed between transformed_b and base: {e}")
            continue

        try:
            intersection_ac = BRepAlgoAPI_Common(transformed_c, base, progress_range).Shape()
        except TypeError as e:
            print(f"Common operation failed between transformed_c and base: {e}")
            continue

        if any(x is None for x in [intersection_ab, intersection_bc, intersection_ac]):
            print("Warning: One or more intersections are None. Skipping this iteration.")
            continue  # 跳过当前循环的其余部分

        try:
            part_a_cut = BRepAlgoAPI_Cut(transformed_a, intersection_ab, progress_range).Shape()
        except TypeError as e:
            print(f"Cut operation failed for part_a: {e}")
            continue

        try:
            part_b_cut = BRepAlgoAPI_Cut(transformed_b, intersection_bc, progress_range).Shape()
        except TypeError as e:
            print(f"Cut operation failed for part_b: {e}")
            continue

        try:
            part_c_cut = BRepAlgoAPI_Cut(transformed_c, intersection_ac, progress_range).Shape()
        except TypeError as e:
            print(f"Cut operation failed for part_c: {e}")
            continue

        try:
            final_model = BRepAlgoAPI_Fuse(base, part_a_cut, progress_range).Shape()
            final_model = BRepAlgoAPI_Fuse(final_model, part_b_cut, progress_range).Shape()
            final_model = BRepAlgoAPI_Fuse(final_model, part_c_cut, progress_range).Shape()
        except TypeError as e:
            print(f"Fuse operation failed during final model assembly: {e}")
            continue

        if mode == 2 or mode == 3:
            try:
                difference_a = BRepAlgoAPI_Fuse(part_a_cut, part_b_cut).Shape()
                difference_a = BRepAlgoAPI_Fuse(difference_a, part_c_cut).Shape()

                transformed_a2 = random_planar_transform_on_contact_face(part_a_cut, base, difference_a)
                difference_a = BRepAlgoAPI_Fuse(difference_a, transformed_a2).Shape()

                transformed_b2 = random_planar_transform_on_contact_face(part_b_cut, base, difference_a)
                difference_a = BRepAlgoAPI_Fuse(difference_a, transformed_b2).Shape()

                transformed_c2 = random_planar_transform_on_contact_face(part_c_cut, base, difference_a)

                final_model = BRepAlgoAPI_Fuse(final_model, transformed_a2).Shape()
                final_model = BRepAlgoAPI_Fuse(final_model, transformed_b2).Shape()
                final_model = BRepAlgoAPI_Fuse(final_model, transformed_c2).Shape()

                if mode == 3:
                    difference_a = BRepAlgoAPI_Fuse(difference_a, transformed_c2).Shape()

                    transformed_a3 = random_planar_transform_on_contact_face(part_a_cut, base, difference_a)
                    difference_a = BRepAlgoAPI_Fuse(difference_a, transformed_a3).Shape()

                    transformed_b3 = random_planar_transform_on_contact_face(part_b_cut, base, difference_a)
                    difference_a = BRepAlgoAPI_Fuse(difference_a, transformed_b3).Shape()

                    transformed_c2 = random_planar_transform_on_contact_face(part_c_cut, base, difference_a)

                    final_model = BRepAlgoAPI_Fuse(final_model, transformed_a2).Shape()
                    final_model = BRepAlgoAPI_Fuse(final_model, transformed_b2).Shape()
                    final_model = BRepAlgoAPI_Fuse(final_model, transformed_c2).Shape()
            except TypeError as e:
                print(f"Advanced fuse operations failed: {e}")
                continue

            mapping = {}
            color_faces([part_a_cut, part_b_cut, part_c_cut, base], final_model, mapping, dizuo)

            if mode==3:
                export_face_to_model_mapping(mapping, rf"C:\Users\20268\Desktop\Project\ProcessData\Data\CBF\addData\json\three_{i}.json")
                write_to_step(final_model, rf"C:\Users\20268\Desktop\Project\ProcessData\Data\CBF\addData\step\three_{i}.step")
            elif mode==2:
                export_face_to_model_mapping(mapping, rf"C:\Users\20268\Desktop\Project\ProcessData\Data\CBF\addData\json\double_{i}.json")
                write_to_step(final_model, rf"C:\Users\20268\Desktop\Project\ProcessData\Data\CBF\addData\step\double_{i}.step")
        elif mode == 1:
            mapping = {}
            color_faces([part_a_cut, part_b_cut, part_c_cut, base], final_model, mapping, dizuo)
            export_face_to_model_mapping(mapping, rf"C:\Users\20268\Desktop\Project\ProcessData\Data\CBF\addData\json\single_{i}.json")
            write_to_step(final_model, rf"C:\Users\20268\Desktop\Project\ProcessData\Data\CBF\addData\step\single_{i}.step")
def export_face_to_model_mapping(mapping, file_path):
    with open(file_path, 'w') as f:
        json.dump(mapping, f, indent=4)

if __name__ == "__main__":
    main()


