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
@str = private unnamed_addr constant [11 x i8] c"----------\00", align 1
@str.1 = private unnamed_addr constant [11 x i8] c"----------\00", align 1

; Function Attrs: nounwind
define dso_local void @accumulate(ptr noundef %0, i32 noundef %1, i32 noundef %2) #0 {
  %4 = alloca ptr, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  store ptr %0, ptr %4, align 4, !tbaa !9
  store i32 %1, ptr %5, align 4, !tbaa !1
  store i32 %2, ptr %6, align 4, !tbaa !1
  %8 = icmp eq i32 %1, 10
  br i1 %8, label %9, label %10

9:                                                ; preds = %3
  br label %40

10:                                               ; preds = %3
  %11 = load ptr, ptr %4, align 4, !tbaa !9
  %12 = load i32, ptr %5, align 4, !tbaa !1
  %13 = getelementptr inbounds [40 x i8], ptr %11, i32 %12
  %14 = getelementptr inbounds [4 x i8], ptr %13, i32 %12
  %15 = load i32, ptr %14, align 4, !tbaa !1
  %16 = load i32, ptr %6, align 4, !tbaa !1
  %17 = add nsw i32 %16, %15
  store i32 %17, ptr %6, align 4, !tbaa !1
  %18 = load ptr, ptr %4, align 4, !tbaa !9
  %19 = load i32, ptr %5, align 4, !tbaa !1
  %20 = getelementptr inbounds [40 x i8], ptr %18, i32 %19
  %21 = getelementptr inbounds [4 x i8], ptr %20, i32 %19
  store i32 %17, ptr %21, align 4, !tbaa !1
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #5
  br label %22

22:                                               ; preds = %32, %10
  %storemerge = phi i32 [ 0, %10 ], [ %34, %32 ]
  store i32 %storemerge, ptr %7, align 4, !tbaa !1
  %23 = icmp slt i32 %storemerge, 10
  br i1 %23, label %25, label %24

24:                                               ; preds = %22
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #5
  br label %35

25:                                               ; preds = %22
  %26 = load i32, ptr %6, align 4, !tbaa !1
  %27 = load ptr, ptr %4, align 4, !tbaa !9
  %28 = load i32, ptr %5, align 4, !tbaa !1
  %29 = getelementptr inbounds [40 x i8], ptr %27, i32 %28
  %30 = load i32, ptr %7, align 4, !tbaa !1
  %31 = getelementptr inbounds [4 x i8], ptr %29, i32 %30
  store i32 %26, ptr %31, align 4, !tbaa !1
  br label %32

32:                                               ; preds = %25
  %33 = load i32, ptr %7, align 4, !tbaa !1
  %34 = add nsw i32 %33, 1
  br label %22, !llvm.loop !12

35:                                               ; preds = %24
  %36 = load ptr, ptr %4, align 4, !tbaa !9
  %37 = load i32, ptr %5, align 4, !tbaa !1
  %38 = add nsw i32 %37, 1
  %39 = load i32, ptr %6, align 4, !tbaa !1
  call void @accumulate(ptr noundef %36, i32 noundef %38, i32 noundef %39)
  br label %40

40:                                               ; preds = %35, %9
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nounwind
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca [10 x [10 x i32]], align 4
  %3 = alloca ptr, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  call void @llvm.memcpy.p0.p0.i32(ptr noundef nonnull align 4 dereferenceable(400) %2, ptr noundef nonnull align 4 dereferenceable(400) @__const.main.m1, i32 400, i1 false)
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #5
  %6 = call ptr @get_matrix([2 x i32] [i32 2, i32 1])
  store ptr %6, ptr %3, align 4, !tbaa !9
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #5
  br label %7

7:                                                ; preds = %20, %0
  %storemerge = phi i32 [ 0, %0 ], [ %22, %20 ]
  store i32 %storemerge, ptr %4, align 4, !tbaa !1
  %8 = icmp slt i32 %storemerge, 10000
  br i1 %8, label %10, label %9

9:                                                ; preds = %7
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #5
  br label %23

10:                                               ; preds = %7
  %11 = load ptr, ptr %3, align 4, !tbaa !9
  call void @mmadd(ptr noundef nonnull %2, ptr noundef %11)
  br i1 false, label %12, label %13

12:                                               ; preds = %10
  br label %19

13:                                               ; preds = %10
  br i1 true, label %14, label %15

14:                                               ; preds = %13
  call void @accumulate(ptr noundef nonnull %2, i32 noundef 0, i32 noundef 1)
  br label %18

15:                                               ; preds = %13
  br i1 poison, label %16, label %17

16:                                               ; preds = %15
  br label %17

17:                                               ; preds = %16, %15
  br label %18

18:                                               ; preds = %17, %14
  br label %19

19:                                               ; preds = %18, %12
  br label %20

20:                                               ; preds = %19
  %21 = load i32, ptr %4, align 4, !tbaa !1
  %22 = add nsw i32 %21, 1
  br label %7, !llvm.loop !14

23:                                               ; preds = %9
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #5
  br label %24

24:                                               ; preds = %29, %23
  %storemerge2 = phi i32 [ 0, %23 ], [ %31, %29 ]
  store i32 %storemerge2, ptr %5, align 4, !tbaa !1
  %25 = icmp slt i32 %storemerge2, 10
  br i1 %25, label %27, label %26

26:                                               ; preds = %24
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #5
  br label %32

27:                                               ; preds = %24
  %28 = load ptr, ptr %3, align 4, !tbaa !9
  call void @mmmul(ptr noundef nonnull %2, ptr noundef %28)
  br label %29

29:                                               ; preds = %27
  %30 = load i32, ptr %5, align 4, !tbaa !1
  %31 = add nsw i32 %30, 1
  br label %24, !llvm.loop !15

32:                                               ; preds = %26
  call void @print_matrix(ptr noundef nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #5
  %33 = load i32, ptr %1, align 4
  ret i32 %33
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i32(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i32, i1 immarg) #2

; Function Attrs: nounwind
define dso_local ptr @get_matrix([2 x i32] %0) #0 {
  %2 = alloca ptr, align 4
  %3 = alloca %struct.config, align 4
  %.elt = extractvalue [2 x i32] %0, 0
  store i32 %.elt, ptr %3, align 4
  %.repack1 = getelementptr inbounds nuw i8, ptr %3, i32 4
  %.elt2 = extractvalue [2 x i32] %0, 1
  store i32 %.elt2, ptr %.repack1, align 4
  %4 = icmp eq i32 %.elt, 1
  br i1 %4, label %5, label %6

5:                                                ; preds = %1
  store ptr @MATRIX_IDENTITY, ptr %2, align 4
  br label %15

6:                                                ; preds = %1
  %7 = load i32, ptr %3, align 4, !tbaa !16
  %8 = icmp eq i32 %7, 2
  br i1 %8, label %9, label %10

9:                                                ; preds = %6
  store ptr @MATRIX_DOUBLE, ptr %2, align 4
  br label %15

10:                                               ; preds = %6
  %11 = load i32, ptr %3, align 4, !tbaa !16
  %12 = icmp eq i32 %11, 3
  br i1 %12, label %13, label %14

13:                                               ; preds = %10
  store ptr @MATRIX_QUAD, ptr %2, align 4
  br label %15

14:                                               ; preds = %10
  store ptr null, ptr %2, align 4
  br label %15

15:                                               ; preds = %14, %13, %9, %5
  %16 = load ptr, ptr %2, align 4
  ret ptr %16
}

; Function Attrs: nounwind
define dso_local void @mmadd(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 4
  %4 = alloca ptr, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store ptr %0, ptr %3, align 4, !tbaa !9
  store ptr %1, ptr %4, align 4, !tbaa !9
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #5
  br label %7

7:                                                ; preds = %30, %2
  %storemerge = phi i32 [ 0, %2 ], [ %32, %30 ]
  store i32 %storemerge, ptr %5, align 4, !tbaa !1
  %8 = icmp slt i32 %storemerge, 10
  br i1 %8, label %10, label %9

9:                                                ; preds = %7
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #5
  br label %33

10:                                               ; preds = %7
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #5
  br label %11

11:                                               ; preds = %26, %10
  %storemerge1 = phi i32 [ 0, %10 ], [ %28, %26 ]
  store i32 %storemerge1, ptr %6, align 4, !tbaa !1
  %12 = icmp slt i32 %storemerge1, 10
  br i1 %12, label %14, label %13

13:                                               ; preds = %11
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #5
  br label %29

14:                                               ; preds = %11
  %15 = load ptr, ptr %4, align 4, !tbaa !9
  %16 = load i32, ptr %5, align 4, !tbaa !1
  %17 = getelementptr inbounds [40 x i8], ptr %15, i32 %16
  %18 = load i32, ptr %6, align 4, !tbaa !1
  %19 = getelementptr inbounds [4 x i8], ptr %17, i32 %18
  %20 = load i32, ptr %19, align 4, !tbaa !1
  %21 = load ptr, ptr %3, align 4, !tbaa !9
  %22 = getelementptr inbounds [40 x i8], ptr %21, i32 %16
  %23 = getelementptr inbounds [4 x i8], ptr %22, i32 %18
  %24 = load i32, ptr %23, align 4, !tbaa !1
  %25 = add nsw i32 %24, %20
  store i32 %25, ptr %23, align 4, !tbaa !1
  br label %26

26:                                               ; preds = %14
  %27 = load i32, ptr %6, align 4, !tbaa !1
  %28 = add nsw i32 %27, 1
  br label %11, !llvm.loop !18

29:                                               ; preds = %13
  br label %30

30:                                               ; preds = %29
  %31 = load i32, ptr %5, align 4, !tbaa !1
  %32 = add nsw i32 %31, 1
  br label %7, !llvm.loop !19

33:                                               ; preds = %9
  ret void
}

; Function Attrs: nounwind
define dso_local void @mmmul(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 4
  %4 = alloca ptr, align 4
  %5 = alloca [10 x [10 x i32]], align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  store ptr %0, ptr %3, align 4, !tbaa !9
  store ptr %1, ptr %4, align 4, !tbaa !9
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #5
  br label %12

12:                                               ; preds = %51, %2
  %storemerge = phi i32 [ 0, %2 ], [ %53, %51 ]
  store i32 %storemerge, ptr %6, align 4, !tbaa !1
  %13 = icmp slt i32 %storemerge, 10
  br i1 %13, label %15, label %14

14:                                               ; preds = %12
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #5
  br label %54

15:                                               ; preds = %12
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #5
  br label %16

16:                                               ; preds = %47, %15
  %storemerge3 = phi i32 [ 0, %15 ], [ %49, %47 ]
  store i32 %storemerge3, ptr %7, align 4, !tbaa !1
  %17 = icmp slt i32 %storemerge3, 10
  br i1 %17, label %19, label %18

18:                                               ; preds = %16
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #5
  br label %50

19:                                               ; preds = %16
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #5
  store i32 0, ptr %8, align 4, !tbaa !1
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #5
  br label %20

20:                                               ; preds = %38, %19
  %storemerge4 = phi i32 [ 0, %19 ], [ %40, %38 ]
  store i32 %storemerge4, ptr %9, align 4, !tbaa !1
  %21 = icmp slt i32 %storemerge4, 10
  br i1 %21, label %23, label %22

22:                                               ; preds = %20
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #5
  br label %41

23:                                               ; preds = %20
  %24 = load ptr, ptr %3, align 4, !tbaa !9
  %25 = load i32, ptr %6, align 4, !tbaa !1
  %26 = getelementptr inbounds [40 x i8], ptr %24, i32 %25
  %27 = load i32, ptr %9, align 4, !tbaa !1
  %28 = getelementptr inbounds [4 x i8], ptr %26, i32 %27
  %29 = load i32, ptr %28, align 4, !tbaa !1
  %30 = load ptr, ptr %4, align 4, !tbaa !9
  %31 = getelementptr inbounds [40 x i8], ptr %30, i32 %27
  %32 = load i32, ptr %7, align 4, !tbaa !1
  %33 = getelementptr inbounds [4 x i8], ptr %31, i32 %32
  %34 = load i32, ptr %33, align 4, !tbaa !1
  %35 = mul nsw i32 %29, %34
  %36 = load i32, ptr %8, align 4, !tbaa !1
  %37 = add nsw i32 %36, %35
  store i32 %37, ptr %8, align 4, !tbaa !1
  br label %38

38:                                               ; preds = %23
  %39 = load i32, ptr %9, align 4, !tbaa !1
  %40 = add nsw i32 %39, 1
  br label %20, !llvm.loop !20

41:                                               ; preds = %22
  %42 = load i32, ptr %8, align 4, !tbaa !1
  %43 = load i32, ptr %6, align 4, !tbaa !1
  %44 = getelementptr inbounds [40 x i8], ptr %5, i32 %43
  %45 = load i32, ptr %7, align 4, !tbaa !1
  %46 = getelementptr inbounds [4 x i8], ptr %44, i32 %45
  store i32 %42, ptr %46, align 4, !tbaa !1
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #5
  br label %47

47:                                               ; preds = %41
  %48 = load i32, ptr %7, align 4, !tbaa !1
  %49 = add nsw i32 %48, 1
  br label %16, !llvm.loop !21

50:                                               ; preds = %18
  br label %51

51:                                               ; preds = %50
  %52 = load i32, ptr %6, align 4, !tbaa !1
  %53 = add nsw i32 %52, 1
  br label %12, !llvm.loop !22

54:                                               ; preds = %14
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #5
  br label %55

55:                                               ; preds = %75, %54
  %storemerge1 = phi i32 [ 0, %54 ], [ %77, %75 ]
  store i32 %storemerge1, ptr %10, align 4, !tbaa !1
  %56 = icmp slt i32 %storemerge1, 10
  br i1 %56, label %58, label %57

57:                                               ; preds = %55
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #5
  br label %78

58:                                               ; preds = %55
  call void @llvm.lifetime.start.p0(ptr nonnull %11) #5
  br label %59

59:                                               ; preds = %71, %58
  %storemerge2 = phi i32 [ 0, %58 ], [ %73, %71 ]
  store i32 %storemerge2, ptr %11, align 4, !tbaa !1
  %60 = icmp slt i32 %storemerge2, 10
  br i1 %60, label %62, label %61

61:                                               ; preds = %59
  call void @llvm.lifetime.end.p0(ptr nonnull %11) #5
  br label %74

62:                                               ; preds = %59
  %63 = load i32, ptr %10, align 4, !tbaa !1
  %64 = getelementptr inbounds [40 x i8], ptr %5, i32 %63
  %65 = load i32, ptr %11, align 4, !tbaa !1
  %66 = getelementptr inbounds [4 x i8], ptr %64, i32 %65
  %67 = load i32, ptr %66, align 4, !tbaa !1
  %68 = load ptr, ptr %3, align 4, !tbaa !9
  %69 = getelementptr inbounds [40 x i8], ptr %68, i32 %63
  %70 = getelementptr inbounds [4 x i8], ptr %69, i32 %65
  store i32 %67, ptr %70, align 4, !tbaa !1
  br label %71

71:                                               ; preds = %62
  %72 = load i32, ptr %11, align 4, !tbaa !1
  %73 = add nsw i32 %72, 1
  br label %59, !llvm.loop !23

74:                                               ; preds = %61
  br label %75

75:                                               ; preds = %74
  %76 = load i32, ptr %10, align 4, !tbaa !1
  %77 = add nsw i32 %76, 1
  br label %55, !llvm.loop !24

78:                                               ; preds = %57
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #5
  ret void
}

; Function Attrs: nounwind
define dso_local void @print_matrix(ptr noundef %0) #0 {
  %2 = alloca ptr, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store ptr %0, ptr %2, align 4, !tbaa !9
  %puts = call i32 @puts(ptr nonnull dereferenceable(1) @str)
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #5
  br label %5

5:                                                ; preds = %24, %1
  %storemerge = phi i32 [ 0, %1 ], [ %26, %24 ]
  store i32 %storemerge, ptr %3, align 4, !tbaa !1
  %6 = icmp slt i32 %storemerge, 10
  br i1 %6, label %8, label %7

7:                                                ; preds = %5
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #5
  br label %27

8:                                                ; preds = %5
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #5
  br label %9

9:                                                ; preds = %20, %8
  %storemerge2 = phi i32 [ 0, %8 ], [ %22, %20 ]
  store i32 %storemerge2, ptr %4, align 4, !tbaa !1
  %10 = icmp slt i32 %storemerge2, 10
  br i1 %10, label %12, label %11

11:                                               ; preds = %9
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #5
  br label %23

12:                                               ; preds = %9
  %13 = load ptr, ptr %2, align 4, !tbaa !9
  %14 = load i32, ptr %3, align 4, !tbaa !1
  %15 = getelementptr inbounds [40 x i8], ptr %13, i32 %14
  %16 = load i32, ptr %4, align 4, !tbaa !1
  %17 = getelementptr inbounds [4 x i8], ptr %15, i32 %16
  %18 = load i32, ptr %17, align 4, !tbaa !1
  %19 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %18) #5
  br label %20

20:                                               ; preds = %12
  %21 = load i32, ptr %4, align 4, !tbaa !1
  %22 = add nsw i32 %21, 1
  br label %9, !llvm.loop !25

23:                                               ; preds = %11
  %putchar = call i32 @putchar(i32 10)
  br label %24

24:                                               ; preds = %23
  %25 = load i32, ptr %3, align 4, !tbaa !1
  %26 = add nsw i32 %25, 1
  br label %5, !llvm.loop !26

27:                                               ; preds = %7
  %puts1 = call i32 @puts(ptr nonnull dereferenceable(1) @str.1)
  ret void
}

declare dso_local i32 @printf(ptr noundef, ...) #3

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) #4

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) #4

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv32" "target-features"="+32bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zcf,+zicsr,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xrivosvisni,-experimental-xrivosvizip,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvkgs,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zifencei,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv32" "target-features"="+32bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zcf,+zicsr,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xrivosvisni,-experimental-xrivosvizip,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvkgs,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zifencei,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" }
attributes #4 = { nofree nounwind }
attributes #5 = { nounwind }

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
!9 = !{!10, !10, i64 0}
!10 = !{!"p1 int", !11, i64 0}
!11 = !{!"any pointer", !3, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
!14 = distinct !{!14, !13}
!15 = distinct !{!15, !13}
!16 = !{!17, !2, i64 0}
!17 = !{!"config", !2, i64 0, !2, i64 4}
!18 = distinct !{!18, !13}
!19 = distinct !{!19, !13}
!20 = distinct !{!20, !13}
!21 = distinct !{!21, !13}
!22 = distinct !{!22, !13}
!23 = distinct !{!23, !13}
!24 = distinct !{!24, !13}
!25 = distinct !{!25, !13}
!26 = distinct !{!26, !13}
