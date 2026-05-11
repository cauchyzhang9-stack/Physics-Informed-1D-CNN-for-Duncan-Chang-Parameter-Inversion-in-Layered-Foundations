!DEC$ FREEFORM
      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD, &
      RPL,DDSDDT,DRPLDE,DRPLDT, &
      STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME, &
      NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT, &
      CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,KSTEP,KINC)

      INCLUDE 'ABA_PARAM.INC'

      CHARACTER*80 CMNAME
      DIMENSION STRESS(NTENS),STATEV(NSTATV), &
      DDSDDE(NTENS,NTENS),DDSDDT(NTENS),DRPLDE(NTENS), &
      STRAN(NTENS),DSTRAN(NTENS),TIME(2),PREDEF(1),DPRED(1), &
      PROPS(NPROPS),COORDS(3),DROT(3,3),DFGRD0(3,3),DFGRD1(3,3)

      DOUBLE PRECISION P_K, P_n, P_Rf, P_c, P_phi, P_Kur, P_Kb, P_m, P_Pa
      DOUBLE PRECISION PRESS, MISES, SIG1_EST, SIG3_EST
      DOUBLE PRECISION S_failure, STRESS_LEVEL
      DOUBLE PRECISION Et, Bt, E_i, Nut, G_mod, ALAMDA
      DOUBLE PRECISION PI, SIN_PHI, COS_PHI, SIG3_LIMIT
      DOUBLE PRECISION TRACE
      INTEGER I, J

      PARAMETER (PI=3.1415926D0)

      ! --- 读取参数 ---
      P_K = PROPS(1); P_n = PROPS(2); P_Rf = PROPS(3)
      P_c = PROPS(4); P_phi = PROPS(5)*PI/180.0D0
      P_Kb = PROPS(7); P_m = PROPS(8); P_Pa = PROPS(9)

      ! --- 1. 使用不变量计算 (不再用 SPRINC) ---
      ! 计算平均应力 p (压为正)
      TRACE = 0.0D0
      DO I = 1, NDI
          TRACE = TRACE + STRESS(I)
      END DO
      PRESS = -TRACE / 3.0D0

      ! 计算 Mises 应力 q
      MISES = 0.0D0
      DO I = 1, NDI
          MISES = MISES + (STRESS(I) + PRESS)**2.0D0
      END DO
      DO I = NDI+1, NTENS
          MISES = MISES + 2.0D0 * STRESS(I)**2.0D0
      END DO
      MISES = SQRT(1.5D0 * MISES)

      ! 反算估算的 Sig3 (用于计算 Ei) 和 Sig1-Sig3 (用于计算 S)
      ! 在三轴压缩下: q = Sig1 - Sig3
      ! Sig3 = p - q/3 (近似估算，足够用于 Duncan-Chang)
      
      SIG3_EST = PRESS - MISES/3.0D0
      
      ! 限制 Sig3 下限 (1 kPa)
      SIG3_LIMIT = SIG3_EST
      IF (SIG3_LIMIT .LT. 1000.0D0) SIG3_LIMIT = 1000.0D0

      ! --- 2. 破坏准则 ---
      SIN_PHI = SIN(P_phi)
      COS_PHI = COS(P_phi)
      
      ! 破坏偏应力 q_f
      S_failure = (2.0D0*P_c*COS_PHI + 2.0D0*SIG3_LIMIT*SIN_PHI)/(1.0D0-SIN_PHI)
      IF (S_failure .LT. 1.0D0) S_failure = 1.0D0

      ! 应力水平 S = q / q_f
      STRESS_LEVEL = MISES / S_failure
      IF (STRESS_LEVEL .GE. 0.95D0) STRESS_LEVEL = 0.95D0

      ! --- 3. 更新模量 ---
      E_i = P_K * P_Pa * ((SIG3_LIMIT/P_Pa)**P_n)
      Et = (1.0D0 - P_Rf * STRESS_LEVEL)**2.0D0 * E_i
      IF (Et .LT. 1.0D5) Et = 1.0D5

      Bt = P_Kb * P_Pa * ((SIG3_LIMIT/P_Pa)**P_m)
      IF (Bt .LT. 1.0D5) Bt = 1.0D5

      ! --- 4. 刚度矩阵与更新 ---
      Nut = (3.0D0*Bt - Et) / (6.0D0*Bt)
      IF (Nut .GT. 0.49D0) Nut = 0.49D0
      IF (Nut .LT. 0.0D0) Nut = 0.0D0

      G_mod = Et / (2.0D0 * (1.0D0 + Nut))
      ALAMDA = Et * Nut / ((1.0D0 + Nut) * (1.0D0 - 2.0D0 * Nut))

      DDSDDE = 0.0D0
      DO I=1, NDI
          DO J=1, NDI
              DDSDDE(I, J) = ALAMDA
          END DO
          DDSDDE(I, I) = ALAMDA + 2.0D0 * G_mod
      END DO
      DO I=NDI+1, NTENS
          DDSDDE(I, I) = G_mod
      END DO

      DO I=1, NTENS
          DO J=1, NTENS
              STRESS(I) = STRESS(I) + DDSDDE(I, J) * DSTRAN(J)
          END DO
      END DO
      
      STATEV(1) = STRESS_LEVEL ! 输出应力水平供检查
      RETURN
      END