; ModuleID = 'merged.ll'
source_filename = "llvm-link"
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32"

%struct.config = type { i32, i32 }

@__const.main.conf = private unnamed_addr constant %struct.config { i32 2, i32 1 }, align 4
@__const.main.m1 = private unnamed_addr constant [10 x [10 x i32]] [[10 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0], [10 x i32] [i32 10, i32 20, i32 30, i32 40, i32 50, i32 0, i32 0, i32 0, i32 0, i32 0], [10 x i32] [i32 100, i32 200, i32 300, i32 400, i32 500, i32 0, i32 0, i32 0, i32 0, i32 0], [10 x i32] [i32 1000, i32 200, i32 300, i32 400, i32 500, i32 0, i32 0, i32 0, i32 0, i32 0], [10 x i32] [i32 100, i32 200, i32 300, i32 400, i32 5000, i32 0, i32 0, i32 0, i32 0, i32 0], [10 x i32] zeroinitializer, [10 x i32] zeroinitializer, [10 x i32] zeroinitializer, [10 x i32] zeroinitializer, [10 x i32] zeroinitializer], align 4
@.str = private unnamed_addr constant [12 x i8] c"----------\0A\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"%i \00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@MATRIX_IDENTITY = internal global <{ <{ i32, [9 x i32] }>, <{ i32, i32, [8 x i32] }>, [10 x i32], [10 x i32], [10 x i32], [10 x i32], [10 x i32], [10 x i32], [10 x i32], [10 x i32] }> <{ <{ i32, [9 x i32] }> <{ i32 1, [9 x i32] zeroinitializer }>, <{ i32, i32, [8 x i32] }> <{ i32 0, i32 1, [8 x i32] zeroinitializer }>, [10 x i32] [i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0], [10 x i32] [i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0], [10 x i32] [i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0], [10 x i32] zeroinitializer, [10 x i32] zeroinitializer, [10 x i32] zeroinitializer, [10 x i32] zeroinitializer, [10 x i32] zeroinitializer }>, align 4
@MATRIX_DOUBLE = internal global <{ <{ i32, [9 x i32] }>, <{ i32, i32, [8 x i32] }>, [10 x i32], [10 x i32], [10 x i32], [10 x i32], [10 x i32], [10 x i32], [10 x i32], [10 x i32] }> <{ <{ i32, [9 x i32] }> <{ i32 2, [9 x i32] zeroinitializer }>, <{ i32, i32, [8 x i32] }> <{ i32 0, i32 2, [8 x i32] zeroinitializer }>, [10 x i32] [i32 0, i32 0, i32 2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0], [10 x i32] [i32 0, i32 0, i32 0, i32 2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0], [10 x i32] [i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0, i32 0, i32 0, i32 0], [10 x i32] zeroinitializer, [10 x i32] zeroinitializer, [10 x i32] zeroinitializer, [10 x i32] zeroinitializer, [10 x i32] zeroinitializer }>, align 4
@MATRIX_QUAD = internal global <{ <{ i32, [9 x i32] }>, <{ i32, i32, [8 x i32] }>, [10 x i32], [10 x i32], [10 x i32], [10 x i32], [10 x i32], [10 x i32], [10 x i32], [10 x i32] }> <{ <{ i32, [9 x i32] }> <{ i32 4, [9 x i32] zeroinitializer }>, <{ i32, i32, [8 x i32] }> <{ i32 0, i32 4, [8 x i32] zeroinitializer }>, [10 x i32] [i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0], [10 x i32] [i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0], [10 x i32] [i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0], [10 x i32] zeroinitializer, [10 x i32] zeroinitializer, [10 x i32] zeroinitializer, [10 x i32] zeroinitializer, [10 x i32] zeroinitializer }>, align 4

; Function Attrs: nounwind
define dso_local void @accumulate(ptr noundef %0, i32 noundef %1, i32 noundef %2) #0 {
  %4 = icmp eq i32 %1, 10
  br i1 %4, label %5, label %6

5:                                                ; preds = %3
  br label %23

6:                                                ; preds = %3
  %7 = getelementptr inbounds [10 x i32], ptr %0, i32 %1
  %8 = getelementptr inbounds [10 x i32], ptr %7, i32 0, i32 %1
  %9 = load i32, ptr %8, align 4, !tbaa !1
  %10 = add nsw i32 %2, %9
  %11 = getelementptr inbounds [10 x i32], ptr %0, i32 %1
  %12 = getelementptr inbounds [10 x i32], ptr %11, i32 0, i32 %1
  store i32 %10, ptr %12, align 4, !tbaa !1
  br label %13

13:                                               ; preds = %19, %6
  %.0 = phi i32 [ 0, %6 ], [ %20, %19 ]
  %14 = icmp slt i32 %.0, 10
  br i1 %14, label %16, label %15

15:                                               ; preds = %13
  br label %21

16:                                               ; preds = %13
  %17 = getelementptr inbounds [10 x i32], ptr %0, i32 %1
  %18 = getelementptr inbounds [10 x i32], ptr %17, i32 0, i32 %.0
  store i32 %10, ptr %18, align 4, !tbaa !1
  br label %19

19:                                               ; preds = %16
  %20 = add nsw i32 %.0, 1
  br label %13, !llvm.loop !9

21:                                               ; preds = %15
  %22 = add nsw i32 %1, 1
  call void @accumulate(ptr noundef %0, i32 noundef %22, i32 noundef %10)
  br label %23

23:                                               ; preds = %21, %5
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nounwind
define dso_local i32 @main() #0 {
  %1 = alloca %struct.config, align 4
  %2 = alloca [10 x [10 x i32]], align 4
  call void @llvm.lifetime.start.p0(ptr %1) #4
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %1, ptr align 4 @__const.main.conf, i32 8, i1 false)
  call void @llvm.lifetime.start.p0(ptr %2) #4
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %2, ptr align 4 @__const.main.m1, i32 400, i1 false)
  %3 = load [2 x i32], ptr %1, align 4
  %4 = call ptr @get_matrix([2 x i32] %3)
  br label %5

5:                                                ; preds = %31, %0
  %.01 = phi i32 [ 0, %0 ], [ %32, %31 ]
  %6 = icmp slt i32 %.01, 10000
  br i1 %6, label %8, label %7

7:                                                ; preds = %5
  br label %33

8:                                                ; preds = %5
  %9 = getelementptr inbounds [10 x [10 x i32]], ptr %2, i32 0, i32 0
  %10 = getelementptr inbounds [10 x [10 x i32]], ptr %4, i32 0, i32 0
  call void @mmadd(ptr noundef %9, ptr noundef %10)
  %11 = getelementptr inbounds nuw %struct.config, ptr %1, i32 0, i32 1
  %12 = load i32, ptr %11, align 4, !tbaa !11
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %14, label %16

14:                                               ; preds = %8
  %15 = getelementptr inbounds [10 x [10 x i32]], ptr %2, i32 0, i32 0
  call void @accumulate(ptr noundef %15, i32 noundef 0, i32 noundef 0)
  br label %30

16:                                               ; preds = %8
  %17 = getelementptr inbounds nuw %struct.config, ptr %1, i32 0, i32 1
  %18 = load i32, ptr %17, align 4, !tbaa !11
  %19 = icmp eq i32 %18, 1
  br i1 %19, label %20, label %22

20:                                               ; preds = %16
  %21 = getelementptr inbounds [10 x [10 x i32]], ptr %2, i32 0, i32 0
  call void @accumulate(ptr noundef %21, i32 noundef 0, i32 noundef 1)
  br label %29

22:                                               ; preds = %16
  %23 = getelementptr inbounds nuw %struct.config, ptr %1, i32 0, i32 1
  %24 = load i32, ptr %23, align 4, !tbaa !11
  %25 = icmp eq i32 %24, 2
  br i1 %25, label %26, label %28

26:                                               ; preds = %22
  %27 = getelementptr inbounds [10 x [10 x i32]], ptr %2, i32 0, i32 0
  call void @accumulate(ptr noundef %27, i32 noundef 0, i32 noundef 2)
  br label %28

28:                                               ; preds = %26, %22
  br label %29

29:                                               ; preds = %28, %20
  br label %30

30:                                               ; preds = %29, %14
  br label %31

31:                                               ; preds = %30
  %32 = add nsw i32 %.01, 1
  br label %5, !llvm.loop !13

33:                                               ; preds = %7
  br label %34

34:                                               ; preds = %40, %33
  %.0 = phi i32 [ 0, %33 ], [ %41, %40 ]
  %35 = icmp slt i32 %.0, 10
  br i1 %35, label %37, label %36

36:                                               ; preds = %34
  br label %42

37:                                               ; preds = %34
  %38 = getelementptr inbounds [10 x [10 x i32]], ptr %2, i32 0, i32 0
  %39 = getelementptr inbounds [10 x [10 x i32]], ptr %4, i32 0, i32 0
  call void @mmmul(ptr noundef %38, ptr noundef %39)
  br label %40

40:                                               ; preds = %37
  %41 = add nsw i32 %.0, 1
  br label %34, !llvm.loop !14

42:                                               ; preds = %36
  %43 = getelementptr inbounds [10 x [10 x i32]], ptr %2, i32 0, i32 0
  call void @print_matrix(ptr noundef %43)
  call void @llvm.lifetime.end.p0(ptr %2) #4
  call void @llvm.lifetime.end.p0(ptr %1) #4
  ret i32 0
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i32(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i32, i1 immarg) #2

; Function Attrs: nounwind
define dso_local ptr @get_matrix([2 x i32] %0) #0 {
  %2 = alloca %struct.config, align 4
  store [2 x i32] %0, ptr %2, align 4
  %3 = getelementptr inbounds nuw %struct.config, ptr %2, i32 0, i32 0
  %4 = load i32, ptr %3, align 4, !tbaa !15
  %5 = icmp eq i32 %4, 1
  br i1 %5, label %6, label %7

6:                                                ; preds = %1
  br label %18

7:                                                ; preds = %1
  %8 = getelementptr inbounds nuw %struct.config, ptr %2, i32 0, i32 0
  %9 = load i32, ptr %8, align 4, !tbaa !15
  %10 = icmp eq i32 %9, 2
  br i1 %10, label %11, label %12

11:                                               ; preds = %7
  br label %18

12:                                               ; preds = %7
  %13 = getelementptr inbounds nuw %struct.config, ptr %2, i32 0, i32 0
  %14 = load i32, ptr %13, align 4, !tbaa !15
  %15 = icmp eq i32 %14, 3
  br i1 %15, label %16, label %17

16:                                               ; preds = %12
  br label %18

17:                                               ; preds = %12
  br label %18

18:                                               ; preds = %17, %16, %11, %6
  %.0 = phi ptr [ @MATRIX_IDENTITY, %6 ], [ @MATRIX_DOUBLE, %11 ], [ @MATRIX_QUAD, %16 ], [ null, %17 ]
  ret ptr %.0
}

; Function Attrs: nounwind
define dso_local void @mmadd(ptr noundef %0, ptr noundef %1) #0 {
  br label %3

3:                                                ; preds = %21, %2
  %.01 = phi i32 [ 0, %2 ], [ %22, %21 ]
  %4 = icmp slt i32 %.01, 10
  br i1 %4, label %6, label %5

5:                                                ; preds = %3
  br label %23

6:                                                ; preds = %3
  br label %7

7:                                                ; preds = %18, %6
  %.0 = phi i32 [ 0, %6 ], [ %19, %18 ]
  %8 = icmp slt i32 %.0, 10
  br i1 %8, label %10, label %9

9:                                                ; preds = %7
  br label %20

10:                                               ; preds = %7
  %11 = getelementptr inbounds [10 x i32], ptr %1, i32 %.01
  %12 = getelementptr inbounds [10 x i32], ptr %11, i32 0, i32 %.0
  %13 = load i32, ptr %12, align 4, !tbaa !1
  %14 = getelementptr inbounds [10 x i32], ptr %0, i32 %.01
  %15 = getelementptr inbounds [10 x i32], ptr %14, i32 0, i32 %.0
  %16 = load i32, ptr %15, align 4, !tbaa !1
  %17 = add nsw i32 %16, %13
  store i32 %17, ptr %15, align 4, !tbaa !1
  br label %18

18:                                               ; preds = %10
  %19 = add nsw i32 %.0, 1
  br label %7, !llvm.loop !16

20:                                               ; preds = %9
  br label %21

21:                                               ; preds = %20
  %22 = add nsw i32 %.01, 1
  br label %3, !llvm.loop !17

23:                                               ; preds = %5
  ret void
}

; Function Attrs: nounwind
define dso_local void @mmmul(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca [10 x [10 x i32]], align 4
  call void @llvm.lifetime.start.p0(ptr %3) #4
  br label %4

4:                                                ; preds = %32, %2
  %.02 = phi i32 [ 0, %2 ], [ %33, %32 ]
  %5 = icmp slt i32 %.02, 10
  br i1 %5, label %7, label %6

6:                                                ; preds = %4
  br label %34

7:                                                ; preds = %4
  br label %8

8:                                                ; preds = %29, %7
  %.03 = phi i32 [ 0, %7 ], [ %30, %29 ]
  %9 = icmp slt i32 %.03, 10
  br i1 %9, label %11, label %10

10:                                               ; preds = %8
  br label %31

11:                                               ; preds = %8
  br label %12

12:                                               ; preds = %24, %11
  %.05 = phi i32 [ 0, %11 ], [ %25, %24 ]
  %.04 = phi i32 [ 0, %11 ], [ %23, %24 ]
  %13 = icmp slt i32 %.05, 10
  br i1 %13, label %15, label %14

14:                                               ; preds = %12
  br label %26

15:                                               ; preds = %12
  %16 = getelementptr inbounds [10 x i32], ptr %0, i32 %.02
  %17 = getelementptr inbounds [10 x i32], ptr %16, i32 0, i32 %.05
  %18 = load i32, ptr %17, align 4, !tbaa !1
  %19 = getelementptr inbounds [10 x i32], ptr %1, i32 %.05
  %20 = getelementptr inbounds [10 x i32], ptr %19, i32 0, i32 %.03
  %21 = load i32, ptr %20, align 4, !tbaa !1
  %22 = mul nsw i32 %18, %21
  %23 = add nsw i32 %.04, %22
  br label %24

24:                                               ; preds = %15
  %25 = add nsw i32 %.05, 1
  br label %12, !llvm.loop !18

26:                                               ; preds = %14
  %27 = getelementptr inbounds [10 x [10 x i32]], ptr %3, i32 0, i32 %.02
  %28 = getelementptr inbounds [10 x i32], ptr %27, i32 0, i32 %.03
  store i32 %.04, ptr %28, align 4, !tbaa !1
  br label %29

29:                                               ; preds = %26
  %30 = add nsw i32 %.03, 1
  br label %8, !llvm.loop !19

31:                                               ; preds = %10
  br label %32

32:                                               ; preds = %31
  %33 = add nsw i32 %.02, 1
  br label %4, !llvm.loop !20

34:                                               ; preds = %6
  br label %35

35:                                               ; preds = %51, %34
  %.01 = phi i32 [ 0, %34 ], [ %52, %51 ]
  %36 = icmp slt i32 %.01, 10
  br i1 %36, label %38, label %37

37:                                               ; preds = %35
  br label %53

38:                                               ; preds = %35
  br label %39

39:                                               ; preds = %48, %38
  %.0 = phi i32 [ 0, %38 ], [ %49, %48 ]
  %40 = icmp slt i32 %.0, 10
  br i1 %40, label %42, label %41

41:                                               ; preds = %39
  br label %50

42:                                               ; preds = %39
  %43 = getelementptr inbounds [10 x [10 x i32]], ptr %3, i32 0, i32 %.01
  %44 = getelementptr inbounds [10 x i32], ptr %43, i32 0, i32 %.0
  %45 = load i32, ptr %44, align 4, !tbaa !1
  %46 = getelementptr inbounds [10 x i32], ptr %0, i32 %.01
  %47 = getelementptr inbounds [10 x i32], ptr %46, i32 0, i32 %.0
  store i32 %45, ptr %47, align 4, !tbaa !1
  br label %48

48:                                               ; preds = %42
  %49 = add nsw i32 %.0, 1
  br label %39, !llvm.loop !21

50:                                               ; preds = %41
  br label %51

51:                                               ; preds = %50
  %52 = add nsw i32 %.01, 1
  br label %35, !llvm.loop !22

53:                                               ; preds = %37
  call void @llvm.lifetime.end.p0(ptr %3) #4
  ret void
}

; Function Attrs: nounwind
define dso_local void @print_matrix(ptr noundef %0) #0 {
  %2 = call i32 (ptr, ...) @printf(ptr noundef @.str)
  br label %3

3:                                                ; preds = %19, %1
  %.01 = phi i32 [ 0, %1 ], [ %20, %19 ]
  %4 = icmp slt i32 %.01, 10
  br i1 %4, label %6, label %5

5:                                                ; preds = %3
  br label %21

6:                                                ; preds = %3
  br label %7

7:                                                ; preds = %15, %6
  %.0 = phi i32 [ 0, %6 ], [ %16, %15 ]
  %8 = icmp slt i32 %.0, 10
  br i1 %8, label %10, label %9

9:                                                ; preds = %7
  br label %17

10:                                               ; preds = %7
  %11 = getelementptr inbounds [10 x i32], ptr %0, i32 %.01
  %12 = getelementptr inbounds [10 x i32], ptr %11, i32 0, i32 %.0
  %13 = load i32, ptr %12, align 4, !tbaa !1
  %14 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef %13)
  br label %15

15:                                               ; preds = %10
  %16 = add nsw i32 %.0, 1
  br label %7, !llvm.loop !23

17:                                               ; preds = %9
  %18 = call i32 (ptr, ...) @printf(ptr noundef @.str.2)
  br label %19

19:                                               ; preds = %17
  %20 = add nsw i32 %.01, 1
  br label %3, !llvm.loop !24

21:                                               ; preds = %5
  %22 = call i32 (ptr, ...) @printf(ptr noundef @.str)
  ret void
}

declare dso_local i32 @printf(ptr noundef, ...) #3

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv32" "target-features"="+32bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zcf,+zicsr,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xrivosvisni,-experimental-xrivosvizip,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvkgs,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zifencei,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv32" "target-features"="+32bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zcf,+zicsr,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xrivosvisni,-experimental-xrivosvizip,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvkgs,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zifencei,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" }
attributes #4 = { nounwind }

!llvm.ident = !{!0}
!llvm.errno.tbaa = !{!1}
!llvm.module.flags = !{!5, !6, !8}

!0 = !{!"clang version 23.0.0git (https://github.com/llvm/llvm-project.git f441746a7b3986d2fcb1f973745b53e33e6c68e4)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{i32 1, !"target-abi", !"ilp32"}
!6 = distinct !{i32 6, !"riscv-isa", !7}
!7 = distinct !{!"rv32i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0_zcf1p0"}
!8 = !{i32 8, !"SmallDataLimit", i32 0}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!12, !2, i64 4}
!12 = !{!"config", !2, i64 0, !2, i64 4}
!13 = distinct !{!13, !10}
!14 = distinct !{!14, !10}
!15 = !{!12, !2, i64 0}
!16 = distinct !{!16, !10}
!17 = distinct !{!17, !10}
!18 = distinct !{!18, !10}
!19 = distinct !{!19, !10}
!20 = distinct !{!20, !10}
!21 = distinct !{!21, !10}
!22 = distinct !{!22, !10}
!23 = distinct !{!23, !10}
!24 = distinct !{!24, !10}
