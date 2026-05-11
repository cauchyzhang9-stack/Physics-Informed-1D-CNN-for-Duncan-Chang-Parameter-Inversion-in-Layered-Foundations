# -*- coding: utf-8 -*-
from abaqus import *
from abaqusConstants import *
from caeModules import *
import regionToolset
import mesh
import os
import csv
import random
import time
from odbAccess import openOdb

# ==============================================================================
# 0. 全局控制 (USER SETTINGS)
# ==============================================================================
# 修改这里来分配任务
TOTAL_RUNS = 1           # 本次任务跑多少组
START_ID   = 1800            # 起始 ID (窗口2设为500, 窗口3设为1000...)
N_CPU      = 2            # 并行时建议设小一点(如2)，留给其他窗口资源
DELETE_ODB = True          
DATA_DIR   = 'Dataset_Final_Unified' # 所有窗口共享同一个文件夹

# ==============================================================================
# 1. 安全写入模块 (核心改进)
# ==============================================================================
def get_log_path():
    if not os.path.exists(DATA_DIR):
        try: os.makedirs(DATA_DIR)
        except: pass
    return os.path.join(DATA_DIR, 'Master_Log.csv')

def init_log_header():
    """初始化表头，带简单的防冲突检查"""
    log_path = get_log_path()
    # 只有文件不存在时才写表头
    if not os.path.exists(log_path):
        try:
            with open(log_path, 'w') as f:
                header = "Run_ID,Given_h1,Given_phi1,Given_phi2,Target_K1,Target_n1,Target_c1,Target_K2,Target_n2,Target_c2,Status\n"
                f.write(header)
        except: 
            pass # 如果报错说明被别的进程抢先创建了，无所谓

def write_log_safe(line_str):
    """防冲突写入函数：如果文件被占用，自动重试"""
    log_path = get_log_path()
    max_retries = 10
    
    for attempt in range(max_retries):
        try:
            with open(log_path, 'a') as f:
                f.write(line_str)
            return True # 写入成功
        except IOError:
            # 写入失败(文件被锁)，等待随机时间后重试
            time.sleep(random.uniform(0.1, 0.5))
    
    print("Warning: Failed to write log for line: " + line_str)
    return False

# ==============================================================================
# 2. 参数范围 (保持一致)
# ==============================================================================
RANGE_h1 = (0.5, 2.0)  
RANGE_phi1 = (20.0, 38.0); RANGE_phi2 = (25.0, 42.0)
RANGE_K1   = (80.0, 400.0); RANGE_n1   = (0.3, 0.7); RANGE_c1   = (5000.0, 35000.0)
RANGE_K2   = (100.0, 500.0); RANGE_n2   = (0.3, 0.7); RANGE_c2   = (5000.0, 50000.0)

FIXED_DENSITY_TOP = 1800.0; FIXED_DENSITY_BOT = 1700.0
FIXED_Rf = 0.80; FIXED_m  = 0.2; FIXED_Pa = 101325.0

# ==============================================================================
# 3. 仿真函数 (引用外挂 UMAT)
# ==============================================================================
def run_simulation(run_id, params):
    h1, K1, n1, c1, phi1, K2, n2, c2, phi2 = params
    job_name = 'Run_%d' % run_id
    model_name = 'Model_%d' % run_id
    subroutine_file = 'duncan_solver.f' # 引用同一个 Fortran 文件
    
    # 清理
    for ext in ['.lck','.odb','.sta','.msg','.log','.com','.sim','.prt','.dat']:
        if os.path.exists(job_name + ext):
            try: os.remove(job_name + ext)
            except: pass

    if model_name in mdb.models: del mdb.models[model_name]
    Mdb()
    mdb.Model(name=model_name)
    myModel = mdb.models[model_name]
    
    # --- 建模过程 (简略展示，与之前完全一致) ---
    soil_width = 12.0; bot_layer_h = 5.0; top_layer_h = h1
    plate_width = 1.0; plate_thick = 0.3; target_disp = -0.15; epsilon = 0.01

    s1 = myModel.ConstrainedSketch(name='s_bot', sheetSize=20.0)
    s1.rectangle(point1=(-soil_width/2.0, -bot_layer_h), point2=(soil_width/2.0, 0.0))
    p_bot = myModel.Part(name='Part-Bot', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    p_bot.BaseShell(sketch=s1)

    s2 = myModel.ConstrainedSketch(name='s_top', sheetSize=20.0)
    s2.rectangle(point1=(-soil_width/2.0, 0.0), point2=(soil_width/2.0, top_layer_h))
    p_top = myModel.Part(name='Part-Top', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    p_top.BaseShell(sketch=s2)

    s3 = myModel.ConstrainedSketch(name='s_plate', sheetSize=5.0)
    s3.rectangle(point1=(-plate_width/2.0, 0.0), point2=(plate_width/2.0, plate_thick))
    p_plate = myModel.Part(name='Part-Plate', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    p_plate.BaseShell(sketch=s3)

    # Materials
    props_top = (K1, n1, FIXED_Rf, c1, phi1, 2.0*K1, 0.6*K1, FIXED_m, FIXED_Pa)
    myModel.Material(name='Mat-Top')
    myModel.materials['Mat-Top'].Density(table=((FIXED_DENSITY_TOP,),))
    myModel.materials['Mat-Top'].UserMaterial(mechanicalConstants=props_top)
    myModel.materials['Mat-Top'].Depvar(n=25)

    props_bot = (K2, n2, FIXED_Rf, c2, phi2, 2.0*K2, 0.6*K2, FIXED_m, FIXED_Pa)
    myModel.Material(name='Mat-Bot')
    myModel.materials['Mat-Bot'].Density(table=((FIXED_DENSITY_BOT,),))
    myModel.materials['Mat-Bot'].UserMaterial(mechanicalConstants=props_bot)
    myModel.materials['Mat-Bot'].Depvar(n=25)

    myModel.Material(name='Mat-Plate')
    myModel.materials['Mat-Plate'].Density(table=((7800.0,),))
    myModel.materials['Mat-Plate'].Elastic(table=((2e11, 0.3),))

    myModel.HomogeneousSolidSection(name='Sec-Top', material='Mat-Top', thickness=1.0)
    myModel.HomogeneousSolidSection(name='Sec-Bot', material='Mat-Bot', thickness=1.0)
    myModel.HomogeneousSolidSection(name='Sec-Plate', material='Mat-Plate', thickness=1.0)

    p_top.SectionAssignment(region=regionToolset.Region(faces=p_top.faces[:]), sectionName='Sec-Top')
    p_bot.SectionAssignment(region=regionToolset.Region(faces=p_bot.faces[:]), sectionName='Sec-Bot')
    p_plate.SectionAssignment(region=regionToolset.Region(faces=p_plate.faces[:]), sectionName='Sec-Plate')

    # Mesh (0.1 size)
    elemType = mesh.ElemType(elemCode=CPE4, elemLibrary=STANDARD)
    for p in [p_top, p_bot, p_plate]:
        p.seedPart(size=0.1) 
        p.setElementType(regions=regionToolset.Region(faces=p.faces[:]), elemTypes=(elemType, ))
        p.generateMesh()

    # Assembly
    a = myModel.rootAssembly; a.DatumCsysByDefault(CARTESIAN)
    inst_bot = a.Instance(name='Inst-Bot', part=p_bot, dependent=ON)
    inst_top = a.Instance(name='Inst-Top', part=p_top, dependent=ON)
    inst_plate = a.Instance(name='Inst-Plate', part=p_plate, dependent=ON)
    a.translate(instanceList=('Inst-Top', ), vector=(0.0, -top_layer_h, 0.0))
    a.translate(instanceList=('Inst-Bot', ), vector=(0.0, -top_layer_h, 0.0))

    # Tie & Contact
    surf_bot = a.Surface(side1Edges=inst_bot.edges.getByBoundingBox(yMin=-top_layer_h-epsilon, yMax=-top_layer_h+epsilon), name='Surf-Bot')
    surf_top = a.Surface(side1Edges=inst_top.edges.getByBoundingBox(yMin=-top_layer_h-epsilon, yMax=-top_layer_h+epsilon), name='Surf-Top')
    myModel.Tie(name='Constraint-Soil', main=surf_bot, secondary=surf_top, positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, thickness=ON)

    myModel.ContactProperty('Prop-Fric')
    myModel.interactionProperties['Prop-Fric'].TangentialBehavior(formulation=PENALTY, table=((0.4,),), fraction=0.005)
    myModel.interactionProperties['Prop-Fric'].NormalBehavior(pressureOverclosure=HARD, allowSeparation=ON)
    surf_p = a.Surface(side1Edges=inst_plate.edges.getByBoundingBox(yMin=-epsilon, yMax=epsilon), name='S-Plate')
    surf_s = a.Surface(side1Edges=inst_top.edges.getByBoundingBox(yMin=-epsilon, yMax=epsilon), name='S-Soil')
    myModel.SurfaceToSurfaceContactStd(name='Int-PlateSoil', createStepName='Initial', main=surf_p, secondary=surf_s, sliding=FINITE, interactionProperty='Prop-Fric')

    # Steps (200 Intervals)
    myModel.StaticStep(name='Step-Grav', previous='Initial', timePeriod=1.0, nlgeom=ON, initialInc=0.01, maxInc=0.1, maxNumInc=10000)
    myModel.StaticStep(name='Step-Load', previous='Step-Grav', timePeriod=1.0, 
                       initialInc=0.001, maxInc=0.005, maxNumInc=10000, nlgeom=ON)
    myModel.fieldOutputRequests['F-Output-1'].setValues(variables=('S', 'U', 'RF'), numIntervals=200)

    # BCs
    bot_y = -(top_layer_h + bot_layer_h)
    e_base = inst_bot.edges.getByBoundingBox(yMin=bot_y-epsilon, yMax=bot_y+epsilon)
    myModel.DisplacementBC(name='BC-Base', createStepName='Initial', region=regionToolset.Region(edges=e_base), u1=SET, u2=SET)
    e_sides = inst_bot.edges.getByBoundingBox(xMin=-soil_width/2-epsilon, xMax=-soil_width/2+epsilon) + \
              inst_top.edges.getByBoundingBox(xMin=-soil_width/2-epsilon, xMax=-soil_width/2+epsilon) + \
              inst_top.edges.getByBoundingBox(xMin=soil_width/2-epsilon, xMax=soil_width/2+epsilon)
    myModel.DisplacementBC(name='BC-Sides', createStepName='Initial', region=regionToolset.Region(edges=e_sides), u1=SET)

    ref_region = regionToolset.Region(edges=inst_plate.edges.getByBoundingBox(yMin=plate_thick-epsilon))
    myModel.DisplacementBC(name='BC-Plate-Float', createStepName='Step-Grav', region=ref_region, u1=SET, u2=UNSET, ur3=SET)
    myModel.boundaryConditions['BC-Plate-Float'].deactivate('Step-Load')
    myModel.DisplacementBC(name='BC-Plate-Push', createStepName='Step-Load', region=ref_region, u1=SET, u2=target_disp, ur3=SET)
    myModel.Gravity(name='Load-G', createStepName='Step-Grav', comp2=-9.8, distributionType=UNIFORM, field='')

    myJob = mdb.Job(name=job_name, model=model_name, type=ANALYSIS, userSubroutine=subroutine_file, numCpus=N_CPU, numDomains=N_CPU, scratch='', resultsFormat=ODB, echoPrint=OFF, modelPrint=OFF, contactPrint=OFF, historyPrint=OFF)
    
    try:
        myJob.submit()
        myJob.waitForCompletion()
        return True
    except:
        return False

# ==============================================================================
# 4. 数据提取 (含归零)
# ==============================================================================
def extract_data(run_id):
    job_name = 'Run_%d' % run_id
    csv_name = 'Data_%d.csv' % run_id
    odb_path = job_name + '.odb'
    
    try:
        if not os.path.exists(odb_path): return False
        odb = openOdb(path=odb_path)
        step = odb.steps['Step-Load']
        
        plate_inst = None
        for k in odb.rootAssembly.instances.keys():
            if 'PLATE' in k.upper(): plate_inst = odb.rootAssembly.instances[k]; break
        if plate_inst is None: odb.close(); return False
        
        data_rows = []
        if len(step.frames) > 0:
            f0 = step.frames[0]
            rf0 = sum([v.data[1] for v in f0.fieldOutputs['RF'].getSubset(region=plate_inst).values])
            u2_0 = sum([v.data[1] for v in f0.fieldOutputs['U'].getSubset(region=plate_inst).values]) / 21
            for frame in step.frames:
                rf = sum([v.data[1] for v in frame.fieldOutputs['RF'].getSubset(region=plate_inst).values])
                u2 = sum([v.data[1] for v in frame.fieldOutputs['U'].getSubset(region=plate_inst).values]) / 21
                p = abs(rf - rf0) / 1.0 / 1000.0
                s = abs(u2 - u2_0) * 1000.0
                if s > 0.00001: data_rows.append((s, p))
        odb.close()
        
        if len(data_rows) > 0:
            with open(os.path.join(DATA_DIR, csv_name), 'wb') as f:
                w = csv.writer(f)
                w.writerow(['Settlement(mm)', 'Pressure(kPa)'])
                w.writerows(data_rows)
            return True
        else: return False
    except: return False

# ==============================================================================
# 5. 主循环 (并行安全版)
# ==============================================================================
if __name__ == '__main__':
    # 尝试初始化表头 (所有进程都会尝试，只要有一个成功就行)
    init_log_header()
    
    print(">>> Starting Parallel Unified Generation (StartID=%d, Total=%d) <<<" % (START_ID, TOTAL_RUNS))
    
    for i in range(START_ID, START_ID + TOTAL_RUNS):
        print("\n--- Run ID: %d ---" % i)
        
        # 1. Random Params
        h1 = random.uniform(*RANGE_h1)
        phi1 = random.uniform(*RANGE_phi1); phi2 = random.uniform(*RANGE_phi2)
        K1 = random.uniform(*RANGE_K1); n1 = random.uniform(*RANGE_n1); c1 = random.uniform(*RANGE_c1)
        K2 = random.uniform(*RANGE_K2); n2 = random.uniform(*RANGE_n2); c2 = random.uniform(*RANGE_c2)
        
        params = (h1, K1, n1, c1, phi1, K2, n2, c2, phi2)
        
        # 2. Run
        sim_success = run_simulation(i, params)
        extract_success = False
        
        if sim_success:
            extract_success = extract_data(i)
            if extract_success:
                print("   -> Success")
                if DELETE_ODB:
                    try: os.remove('Run_%d.odb' % i); print("   -> ODB Deleted")
                    except: pass
            else:
                print("   -> Extract Failed")
        else:
            print("   -> Sim Failed")
        
        # 3. Log (Safe Write)
        final_status = "Success" if (sim_success and extract_success) else "Failed"
        line = "%d,%.3f,%.1f,%.1f,%.1f,%.3f,%.1f,%.1f,%.3f,%.1f,%s\n" % \
               (i, h1, phi1, phi2, K1, n1, c1, K2, n2, c2, final_status)
        
        # 调用安全写入函数
        write_log_safe(line)

    print("\n>>> Batch Completed!")