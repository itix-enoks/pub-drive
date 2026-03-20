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
  %4 = alloca ptr, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  store ptr %0, ptr %4, align 4, !tbaa !9
  store i32 %1, ptr %5, align 4, !tbaa !1
  store i32 %2, ptr %6, align 4, !tbaa !1
  %8 = load i32, ptr %5, align 4, !tbaa !1
  %9 = icmp eq i32 %8, 10
  br i1 %9, label %10, label %11

10:                                               ; preds = %3
  br label %106

11:                                               ; preds = %3
  %12 = load ptr, ptr %4, align 4, !tbaa !9
  %13 = load i32, ptr %5, align 4, !tbaa !1
  %14 = getelementptr inbounds [10 x i32], ptr %12, i32 %13
  %15 = load i32, ptr %5, align 4, !tbaa !1
  %16 = getelementptr inbounds [10 x i32], ptr %14, i32 0, i32 %15
  %17 = load i32, ptr %16, align 4, !tbaa !1
  %18 = load i32, ptr %6, align 4, !tbaa !1
  %19 = add nsw i32 %18, %17
  store i32 %19, ptr %6, align 4, !tbaa !1
  %20 = load i32, ptr %6, align 4, !tbaa !1
  %21 = load ptr, ptr %4, align 4, !tbaa !9
  %22 = load i32, ptr %5, align 4, !tbaa !1
  %23 = getelementptr inbounds [10 x i32], ptr %21, i32 %22
  %24 = load i32, ptr %5, align 4, !tbaa !1
  %25 = getelementptr inbounds [10 x i32], ptr %23, i32 0, i32 %24
  store i32 %20, ptr %25, align 4, !tbaa !1
  call void @llvm.lifetime.start.p0(ptr %7) #4
  store i32 0, ptr %7, align 4, !tbaa !1
  br label %26

26:                                               ; preds = %99, %11
  %27 = load i32, ptr %7, align 4, !tbaa !1
  %28 = icmp slt i32 %27, 10
  br i1 %28, label %30, label %29

29:                                               ; preds = %92, %85, %78, %71, %64, %57, %50, %43, %36, %26
  call void @llvm.lifetime.end.p0(ptr %7) #4
  br label %101

30:                                               ; preds = %26
  %31 = load i32, ptr %6, align 4, !tbaa !1
  %32 = load ptr, ptr %4, align 4, !tbaa !9
  %33 = load i32, ptr %5, align 4, !tbaa !1
  %34 = getelementptr inbounds [10 x i32], ptr %32, i32 %33
  %35 = getelementptr inbounds [10 x i32], ptr %34, i32 0, i32 %27
  store i32 %31, ptr %35, align 4, !tbaa !1
  br label %36

36:                                               ; preds = %30
  %37 = add nsw i32 %27, 1
  store i32 %37, ptr %7, align 4, !tbaa !1
  %38 = load i32, ptr %7, align 4, !tbaa !1
  %39 = icmp slt i32 %38, 10
  br i1 %39, label %40, label %29

40:                                               ; preds = %36
  %41 = getelementptr inbounds [10 x i32], ptr %32, i32 %33
  %42 = getelementptr inbounds [10 x i32], ptr %41, i32 0, i32 %38
  store i32 %31, ptr %42, align 4, !tbaa !1
  br label %43

43:                                               ; preds = %40
  %44 = add nsw i32 %38, 1
  store i32 %44, ptr %7, align 4, !tbaa !1
  %45 = load i32, ptr %7, align 4, !tbaa !1
  %46 = icmp slt i32 %45, 10
  br i1 %46, label %47, label %29

47:                                               ; preds = %43
  %48 = getelementptr inbounds [10 x i32], ptr %32, i32 %33
  %49 = getelementptr inbounds [10 x i32], ptr %48, i32 0, i32 %45
  store i32 %31, ptr %49, align 4, !tbaa !1
  br label %50

50:                                               ; preds = %47
  %51 = add nsw i32 %45, 1
  store i32 %51, ptr %7, align 4, !tbaa !1
  %52 = load i32, ptr %7, align 4, !tbaa !1
  %53 = icmp slt i32 %52, 10
  br i1 %53, label %54, label %29

54:                                               ; preds = %50
  %55 = getelementptr inbounds [10 x i32], ptr %32, i32 %33
  %56 = getelementptr inbounds [10 x i32], ptr %55, i32 0, i32 %52
  store i32 %31, ptr %56, align 4, !tbaa !1
  br label %57

57:                                               ; preds = %54
  %58 = add nsw i32 %52, 1
  store i32 %58, ptr %7, align 4, !tbaa !1
  %59 = load i32, ptr %7, align 4, !tbaa !1
  %60 = icmp slt i32 %59, 10
  br i1 %60, label %61, label %29

61:                                               ; preds = %57
  %62 = getelementptr inbounds [10 x i32], ptr %32, i32 %33
  %63 = getelementptr inbounds [10 x i32], ptr %62, i32 0, i32 %59
  store i32 %31, ptr %63, align 4, !tbaa !1
  br label %64

64:                                               ; preds = %61
  %65 = add nsw i32 %59, 1
  store i32 %65, ptr %7, align 4, !tbaa !1
  %66 = load i32, ptr %7, align 4, !tbaa !1
  %67 = icmp slt i32 %66, 10
  br i1 %67, label %68, label %29

68:                                               ; preds = %64
  %69 = getelementptr inbounds [10 x i32], ptr %32, i32 %33
  %70 = getelementptr inbounds [10 x i32], ptr %69, i32 0, i32 %66
  store i32 %31, ptr %70, align 4, !tbaa !1
  br label %71

71:                                               ; preds = %68
  %72 = add nsw i32 %66, 1
  store i32 %72, ptr %7, align 4, !tbaa !1
  %73 = load i32, ptr %7, align 4, !tbaa !1
  %74 = icmp slt i32 %73, 10
  br i1 %74, label %75, label %29

75:                                               ; preds = %71
  %76 = getelementptr inbounds [10 x i32], ptr %32, i32 %33
  %77 = getelementptr inbounds [10 x i32], ptr %76, i32 0, i32 %73
  store i32 %31, ptr %77, align 4, !tbaa !1
  br label %78

78:                                               ; preds = %75
  %79 = add nsw i32 %73, 1
  store i32 %79, ptr %7, align 4, !tbaa !1
  %80 = load i32, ptr %7, align 4, !tbaa !1
  %81 = icmp slt i32 %80, 10
  br i1 %81, label %82, label %29

82:                                               ; preds = %78
  %83 = getelementptr inbounds [10 x i32], ptr %32, i32 %33
  %84 = getelementptr inbounds [10 x i32], ptr %83, i32 0, i32 %80
  store i32 %31, ptr %84, align 4, !tbaa !1
  br label %85

85:                                               ; preds = %82
  %86 = add nsw i32 %80, 1
  store i32 %86, ptr %7, align 4, !tbaa !1
  %87 = load i32, ptr %7, align 4, !tbaa !1
  %88 = icmp slt i32 %87, 10
  br i1 %88, label %89, label %29

89:                                               ; preds = %85
  %90 = getelementptr inbounds [10 x i32], ptr %32, i32 %33
  %91 = getelementptr inbounds [10 x i32], ptr %90, i32 0, i32 %87
  store i32 %31, ptr %91, align 4, !tbaa !1
  br label %92

92:                                               ; preds = %89
  %93 = add nsw i32 %87, 1
  store i32 %93, ptr %7, align 4, !tbaa !1
  %94 = load i32, ptr %7, align 4, !tbaa !1
  %95 = icmp slt i32 %94, 10
  br i1 %95, label %96, label %29

96:                                               ; preds = %92
  %97 = getelementptr inbounds [10 x i32], ptr %32, i32 %33
  %98 = getelementptr inbounds [10 x i32], ptr %97, i32 0, i32 %94
  store i32 %31, ptr %98, align 4, !tbaa !1
  br label %99

99:                                               ; preds = %96
  %100 = add nsw i32 %94, 1
  store i32 %100, ptr %7, align 4, !tbaa !1
  br label %26, !llvm.loop !12

101:                                              ; preds = %29
  %102 = load ptr, ptr %4, align 4, !tbaa !9
  %103 = load i32, ptr %5, align 4, !tbaa !1
  %104 = add nsw i32 %103, 1
  %105 = load i32, ptr %6, align 4, !tbaa !1
  call void @accumulate(ptr noundef %102, i32 noundef %104, i32 noundef %105)
  br label %106

106:                                              ; preds = %101, %10
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nounwind
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.config, align 4
  %3 = alloca [10 x [10 x i32]], align 4
  %4 = alloca ptr, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  call void @llvm.lifetime.start.p0(ptr %2) #4
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %2, ptr align 4 @__const.main.conf, i32 8, i1 false)
  call void @llvm.lifetime.start.p0(ptr %3) #4
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %3, ptr align 4 @__const.main.m1, i32 400, i1 false)
  call void @llvm.lifetime.start.p0(ptr %4) #4
  %7 = load [2 x i32], ptr %2, align 4
  %8 = call ptr @get_matrix([2 x i32] %7)
  store ptr %8, ptr %4, align 4, !tbaa !9
  call void @llvm.lifetime.start.p0(ptr %5) #4
  store i32 0, ptr %5, align 4, !tbaa !1
  br label %9

9:                                                ; preds = %37, %0
  %10 = load i32, ptr %5, align 4, !tbaa !1
  %11 = icmp slt i32 %10, 10000
  br i1 %11, label %13, label %12

12:                                               ; preds = %9
  call void @llvm.lifetime.end.p0(ptr %5) #4
  br label %40

13:                                               ; preds = %9
  %14 = getelementptr inbounds [10 x [10 x i32]], ptr %3, i32 0, i32 0
  %15 = load ptr, ptr %4, align 4, !tbaa !9
  %16 = getelementptr inbounds [10 x [10 x i32]], ptr %15, i32 0, i32 0
  call void @mmadd(ptr noundef %14, ptr noundef %16)
  %17 = getelementptr inbounds nuw %struct.config, ptr %2, i32 0, i32 1
  %18 = load i32, ptr %17, align 4, !tbaa !15
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %20, label %22

20:                                               ; preds = %13
  %21 = getelementptr inbounds [10 x [10 x i32]], ptr %3, i32 0, i32 0
  call void @accumulate(ptr noundef %21, i32 noundef 0, i32 noundef 0)
  br label %36

22:                                               ; preds = %13
  %23 = getelementptr inbounds nuw %struct.config, ptr %2, i32 0, i32 1
  %24 = load i32, ptr %23, align 4, !tbaa !15
  %25 = icmp eq i32 %24, 1
  br i1 %25, label %26, label %28

26:                                               ; preds = %22
  %27 = getelementptr inbounds [10 x [10 x i32]], ptr %3, i32 0, i32 0
  call void @accumulate(ptr noundef %27, i32 noundef 0, i32 noundef 1)
  br label %35

28:                                               ; preds = %22
  %29 = getelementptr inbounds nuw %struct.config, ptr %2, i32 0, i32 1
  %30 = load i32, ptr %29, align 4, !tbaa !15
  %31 = icmp eq i32 %30, 2
  br i1 %31, label %32, label %34

32:                                               ; preds = %28
  %33 = getelementptr inbounds [10 x [10 x i32]], ptr %3, i32 0, i32 0
  call void @accumulate(ptr noundef %33, i32 noundef 0, i32 noundef 2)
  br label %34

34:                                               ; preds = %32, %28
  br label %35

35:                                               ; preds = %34, %26
  br label %36

36:                                               ; preds = %35, %20
  br label %37

37:                                               ; preds = %36
  %38 = load i32, ptr %5, align 4, !tbaa !1
  %39 = add nsw i32 %38, 1
  store i32 %39, ptr %5, align 4, !tbaa !1
  br label %9, !llvm.loop !17

40:                                               ; preds = %12
  call void @llvm.lifetime.start.p0(ptr %6) #4
  store i32 0, ptr %6, align 4, !tbaa !1
  br label %41

41:                                               ; preds = %92, %40
  %42 = load i32, ptr %6, align 4, !tbaa !1
  %43 = icmp slt i32 %42, 10
  br i1 %43, label %45, label %44

44:                                               ; preds = %87, %82, %77, %72, %67, %62, %57, %52, %47, %41
  call void @llvm.lifetime.end.p0(ptr %6) #4
  br label %94

45:                                               ; preds = %41
  %46 = load ptr, ptr %4, align 4, !tbaa !9
  call void @mmmul(ptr noundef %3, ptr noundef %46)
  br label %47

47:                                               ; preds = %45
  %48 = add nsw i32 %42, 1
  store i32 %48, ptr %6, align 4, !tbaa !1
  %49 = load i32, ptr %6, align 4, !tbaa !1
  %50 = icmp slt i32 %49, 10
  br i1 %50, label %51, label %44

51:                                               ; preds = %47
  call void @mmmul(ptr noundef %3, ptr noundef %46)
  br label %52

52:                                               ; preds = %51
  %53 = add nsw i32 %49, 1
  store i32 %53, ptr %6, align 4, !tbaa !1
  %54 = load i32, ptr %6, align 4, !tbaa !1
  %55 = icmp slt i32 %54, 10
  br i1 %55, label %56, label %44

56:                                               ; preds = %52
  call void @mmmul(ptr noundef %3, ptr noundef %46)
  br label %57

57:                                               ; preds = %56
  %58 = add nsw i32 %54, 1
  store i32 %58, ptr %6, align 4, !tbaa !1
  %59 = load i32, ptr %6, align 4, !tbaa !1
  %60 = icmp slt i32 %59, 10
  br i1 %60, label %61, label %44

61:                                               ; preds = %57
  call void @mmmul(ptr noundef %3, ptr noundef %46)
  br label %62

62:                                               ; preds = %61
  %63 = add nsw i32 %59, 1
  store i32 %63, ptr %6, align 4, !tbaa !1
  %64 = load i32, ptr %6, align 4, !tbaa !1
  %65 = icmp slt i32 %64, 10
  br i1 %65, label %66, label %44

66:                                               ; preds = %62
  call void @mmmul(ptr noundef %3, ptr noundef %46)
  br label %67

67:                                               ; preds = %66
  %68 = add nsw i32 %64, 1
  store i32 %68, ptr %6, align 4, !tbaa !1
  %69 = load i32, ptr %6, align 4, !tbaa !1
  %70 = icmp slt i32 %69, 10
  br i1 %70, label %71, label %44

71:                                               ; preds = %67
  call void @mmmul(ptr noundef %3, ptr noundef %46)
  br label %72

72:                                               ; preds = %71
  %73 = add nsw i32 %69, 1
  store i32 %73, ptr %6, align 4, !tbaa !1
  %74 = load i32, ptr %6, align 4, !tbaa !1
  %75 = icmp slt i32 %74, 10
  br i1 %75, label %76, label %44

76:                                               ; preds = %72
  call void @mmmul(ptr noundef %3, ptr noundef %46)
  br label %77

77:                                               ; preds = %76
  %78 = add nsw i32 %74, 1
  store i32 %78, ptr %6, align 4, !tbaa !1
  %79 = load i32, ptr %6, align 4, !tbaa !1
  %80 = icmp slt i32 %79, 10
  br i1 %80, label %81, label %44

81:                                               ; preds = %77
  call void @mmmul(ptr noundef %3, ptr noundef %46)
  br label %82

82:                                               ; preds = %81
  %83 = add nsw i32 %79, 1
  store i32 %83, ptr %6, align 4, !tbaa !1
  %84 = load i32, ptr %6, align 4, !tbaa !1
  %85 = icmp slt i32 %84, 10
  br i1 %85, label %86, label %44

86:                                               ; preds = %82
  call void @mmmul(ptr noundef %3, ptr noundef %46)
  br label %87

87:                                               ; preds = %86
  %88 = add nsw i32 %84, 1
  store i32 %88, ptr %6, align 4, !tbaa !1
  %89 = load i32, ptr %6, align 4, !tbaa !1
  %90 = icmp slt i32 %89, 10
  br i1 %90, label %91, label %44

91:                                               ; preds = %87
  call void @mmmul(ptr noundef %3, ptr noundef %46)
  br label %92

92:                                               ; preds = %91
  %93 = add nsw i32 %89, 1
  store i32 %93, ptr %6, align 4, !tbaa !1
  br label %41, !llvm.loop !18

94:                                               ; preds = %44
  %95 = getelementptr inbounds [10 x [10 x i32]], ptr %3, i32 0, i32 0
  call void @print_matrix(ptr noundef %95)
  call void @llvm.lifetime.end.p0(ptr %4) #4
  call void @llvm.lifetime.end.p0(ptr %3) #4
  call void @llvm.lifetime.end.p0(ptr %2) #4
  %96 = load i32, ptr %1, align 4
  ret i32 %96
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i32(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i32, i1 immarg) #2

; Function Attrs: nounwind
define dso_local ptr @get_matrix([2 x i32] %0) #0 {
  %2 = alloca ptr, align 4
  %3 = alloca %struct.config, align 4
  store [2 x i32] %0, ptr %3, align 4
  %4 = getelementptr inbounds nuw %struct.config, ptr %3, i32 0, i32 0
  %5 = load i32, ptr %4, align 4, !tbaa !19
  %6 = icmp eq i32 %5, 1
  br i1 %6, label %7, label %8

7:                                                ; preds = %1
  store ptr @MATRIX_IDENTITY, ptr %2, align 4
  br label %19

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw %struct.config, ptr %3, i32 0, i32 0
  %10 = load i32, ptr %9, align 4, !tbaa !19
  %11 = icmp eq i32 %10, 2
  br i1 %11, label %12, label %13

12:                                               ; preds = %8
  store ptr @MATRIX_DOUBLE, ptr %2, align 4
  br label %19

13:                                               ; preds = %8
  %14 = getelementptr inbounds nuw %struct.config, ptr %3, i32 0, i32 0
  %15 = load i32, ptr %14, align 4, !tbaa !19
  %16 = icmp eq i32 %15, 3
  br i1 %16, label %17, label %18

17:                                               ; preds = %13
  store ptr @MATRIX_QUAD, ptr %2, align 4
  br label %19

18:                                               ; preds = %13
  store ptr null, ptr %2, align 4
  br label %19

19:                                               ; preds = %18, %17, %12, %7
  %20 = load ptr, ptr %2, align 4
  ret ptr %20
}

; Function Attrs: nounwind
define dso_local void @mmadd(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 4
  %4 = alloca ptr, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  store ptr %0, ptr %3, align 4, !tbaa !9
  store ptr %1, ptr %4, align 4, !tbaa !9
  call void @llvm.lifetime.start.p0(ptr %5) #4
  store i32 0, ptr %5, align 4, !tbaa !1
  br label %8

8:                                                ; preds = %35, %2
  %9 = load i32, ptr %5, align 4, !tbaa !1
  %10 = icmp slt i32 %9, 10
  br i1 %10, label %12, label %11

11:                                               ; preds = %8
  store i32 2, ptr %6, align 4
  call void @llvm.lifetime.end.p0(ptr %5) #4
  br label %38

12:                                               ; preds = %8
  call void @llvm.lifetime.start.p0(ptr %7) #4
  store i32 0, ptr %7, align 4, !tbaa !1
  br label %13

13:                                               ; preds = %31, %12
  %14 = load i32, ptr %7, align 4, !tbaa !1
  %15 = icmp slt i32 %14, 10
  br i1 %15, label %17, label %16

16:                                               ; preds = %13
  store i32 5, ptr %6, align 4
  call void @llvm.lifetime.end.p0(ptr %7) #4
  br label %34

17:                                               ; preds = %13
  %18 = load ptr, ptr %4, align 4, !tbaa !9
  %19 = load i32, ptr %5, align 4, !tbaa !1
  %20 = getelementptr inbounds [10 x i32], ptr %18, i32 %19
  %21 = load i32, ptr %7, align 4, !tbaa !1
  %22 = getelementptr inbounds [10 x i32], ptr %20, i32 0, i32 %21
  %23 = load i32, ptr %22, align 4, !tbaa !1
  %24 = load ptr, ptr %3, align 4, !tbaa !9
  %25 = load i32, ptr %5, align 4, !tbaa !1
  %26 = getelementptr inbounds [10 x i32], ptr %24, i32 %25
  %27 = load i32, ptr %7, align 4, !tbaa !1
  %28 = getelementptr inbounds [10 x i32], ptr %26, i32 0, i32 %27
  %29 = load i32, ptr %28, align 4, !tbaa !1
  %30 = add nsw i32 %29, %23
  store i32 %30, ptr %28, align 4, !tbaa !1
  br label %31

31:                                               ; preds = %17
  %32 = load i32, ptr %7, align 4, !tbaa !1
  %33 = add nsw i32 %32, 1
  store i32 %33, ptr %7, align 4, !tbaa !1
  br label %13, !llvm.loop !20

34:                                               ; preds = %16
  br label %35

35:                                               ; preds = %34
  %36 = load i32, ptr %5, align 4, !tbaa !1
  %37 = add nsw i32 %36, 1
  store i32 %37, ptr %5, align 4, !tbaa !1
  br label %8, !llvm.loop !21

38:                                               ; preds = %11
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
  %12 = alloca i32, align 4
  store ptr %0, ptr %3, align 4, !tbaa !9
  store ptr %1, ptr %4, align 4, !tbaa !9
  call void @llvm.lifetime.start.p0(ptr %5) #4
  call void @llvm.lifetime.start.p0(ptr %6) #4
  store i32 0, ptr %6, align 4, !tbaa !1
  br label %13

13:                                               ; preds = %56, %2
  %14 = load i32, ptr %6, align 4, !tbaa !1
  %15 = icmp slt i32 %14, 10
  br i1 %15, label %17, label %16

16:                                               ; preds = %13
  store i32 2, ptr %7, align 4
  call void @llvm.lifetime.end.p0(ptr %6) #4
  br label %59

17:                                               ; preds = %13
  call void @llvm.lifetime.start.p0(ptr %8) #4
  store i32 0, ptr %8, align 4, !tbaa !1
  br label %18

18:                                               ; preds = %52, %17
  %19 = load i32, ptr %8, align 4, !tbaa !1
  %20 = icmp slt i32 %19, 10
  br i1 %20, label %22, label %21

21:                                               ; preds = %18
  store i32 5, ptr %7, align 4
  call void @llvm.lifetime.end.p0(ptr %8) #4
  br label %55

22:                                               ; preds = %18
  call void @llvm.lifetime.start.p0(ptr %9) #4
  store i32 0, ptr %9, align 4, !tbaa !1
  call void @llvm.lifetime.start.p0(ptr %10) #4
  store i32 0, ptr %10, align 4, !tbaa !1
  br label %23

23:                                               ; preds = %43, %22
  %24 = load i32, ptr %10, align 4, !tbaa !1
  %25 = icmp slt i32 %24, 10
  br i1 %25, label %27, label %26

26:                                               ; preds = %23
  store i32 8, ptr %7, align 4
  call void @llvm.lifetime.end.p0(ptr %10) #4
  br label %46

27:                                               ; preds = %23
  %28 = load ptr, ptr %3, align 4, !tbaa !9
  %29 = load i32, ptr %6, align 4, !tbaa !1
  %30 = getelementptr inbounds [10 x i32], ptr %28, i32 %29
  %31 = load i32, ptr %10, align 4, !tbaa !1
  %32 = getelementptr inbounds [10 x i32], ptr %30, i32 0, i32 %31
  %33 = load i32, ptr %32, align 4, !tbaa !1
  %34 = load ptr, ptr %4, align 4, !tbaa !9
  %35 = load i32, ptr %10, align 4, !tbaa !1
  %36 = getelementptr inbounds [10 x i32], ptr %34, i32 %35
  %37 = load i32, ptr %8, align 4, !tbaa !1
  %38 = getelementptr inbounds [10 x i32], ptr %36, i32 0, i32 %37
  %39 = load i32, ptr %38, align 4, !tbaa !1
  %40 = mul nsw i32 %33, %39
  %41 = load i32, ptr %9, align 4, !tbaa !1
  %42 = add nsw i32 %41, %40
  store i32 %42, ptr %9, align 4, !tbaa !1
  br label %43

43:                                               ; preds = %27
  %44 = load i32, ptr %10, align 4, !tbaa !1
  %45 = add nsw i32 %44, 1
  store i32 %45, ptr %10, align 4, !tbaa !1
  br label %23, !llvm.loop !22

46:                                               ; preds = %26
  %47 = load i32, ptr %9, align 4, !tbaa !1
  %48 = load i32, ptr %6, align 4, !tbaa !1
  %49 = getelementptr inbounds [10 x [10 x i32]], ptr %5, i32 0, i32 %48
  %50 = load i32, ptr %8, align 4, !tbaa !1
  %51 = getelementptr inbounds [10 x i32], ptr %49, i32 0, i32 %50
  store i32 %47, ptr %51, align 4, !tbaa !1
  call void @llvm.lifetime.end.p0(ptr %9) #4
  br label %52

52:                                               ; preds = %46
  %53 = load i32, ptr %8, align 4, !tbaa !1
  %54 = add nsw i32 %53, 1
  store i32 %54, ptr %8, align 4, !tbaa !1
  br label %18, !llvm.loop !23

55:                                               ; preds = %21
  br label %56

56:                                               ; preds = %55
  %57 = load i32, ptr %6, align 4, !tbaa !1
  %58 = add nsw i32 %57, 1
  store i32 %58, ptr %6, align 4, !tbaa !1
  br label %13, !llvm.loop !24

59:                                               ; preds = %16
  call void @llvm.lifetime.start.p0(ptr %11) #4
  store i32 0, ptr %11, align 4, !tbaa !1
  br label %60

60:                                               ; preds = %84, %59
  %61 = load i32, ptr %11, align 4, !tbaa !1
  %62 = icmp slt i32 %61, 10
  br i1 %62, label %64, label %63

63:                                               ; preds = %60
  store i32 11, ptr %7, align 4
  call void @llvm.lifetime.end.p0(ptr %11) #4
  br label %87

64:                                               ; preds = %60
  call void @llvm.lifetime.start.p0(ptr %12) #4
  store i32 0, ptr %12, align 4, !tbaa !1
  br label %65

65:                                               ; preds = %80, %64
  %66 = load i32, ptr %12, align 4, !tbaa !1
  %67 = icmp slt i32 %66, 10
  br i1 %67, label %69, label %68

68:                                               ; preds = %65
  store i32 14, ptr %7, align 4
  call void @llvm.lifetime.end.p0(ptr %12) #4
  br label %83

69:                                               ; preds = %65
  %70 = load i32, ptr %11, align 4, !tbaa !1
  %71 = getelementptr inbounds [10 x [10 x i32]], ptr %5, i32 0, i32 %70
  %72 = load i32, ptr %12, align 4, !tbaa !1
  %73 = getelementptr inbounds [10 x i32], ptr %71, i32 0, i32 %72
  %74 = load i32, ptr %73, align 4, !tbaa !1
  %75 = load ptr, ptr %3, align 4, !tbaa !9
  %76 = load i32, ptr %11, align 4, !tbaa !1
  %77 = getelementptr inbounds [10 x i32], ptr %75, i32 %76
  %78 = load i32, ptr %12, align 4, !tbaa !1
  %79 = getelementptr inbounds [10 x i32], ptr %77, i32 0, i32 %78
  store i32 %74, ptr %79, align 4, !tbaa !1
  br label %80

80:                                               ; preds = %69
  %81 = load i32, ptr %12, align 4, !tbaa !1
  %82 = add nsw i32 %81, 1
  store i32 %82, ptr %12, align 4, !tbaa !1
  br label %65, !llvm.loop !25

83:                                               ; preds = %68
  br label %84

84:                                               ; preds = %83
  %85 = load i32, ptr %11, align 4, !tbaa !1
  %86 = add nsw i32 %85, 1
  store i32 %86, ptr %11, align 4, !tbaa !1
  br label %60, !llvm.loop !26

87:                                               ; preds = %63
  call void @llvm.lifetime.end.p0(ptr %5) #4
  ret void
}

; Function Attrs: nounwind
define dso_local void @print_matrix(ptr noundef %0) #0 {
  %2 = alloca ptr, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store ptr %0, ptr %2, align 4, !tbaa !9
  %6 = call i32 (ptr, ...) @printf(ptr noundef @.str)
  call void @llvm.lifetime.start.p0(ptr %3) #4
  store i32 0, ptr %3, align 4, !tbaa !1
  br label %7

7:                                                ; preds = %108, %1
  %8 = load i32, ptr %3, align 4, !tbaa !1
  %9 = icmp slt i32 %8, 10
  br i1 %9, label %11, label %10

10:                                               ; preds = %7
  store i32 2, ptr %4, align 4
  call void @llvm.lifetime.end.p0(ptr %3) #4
  br label %111

11:                                               ; preds = %7
  call void @llvm.lifetime.start.p0(ptr %5) #4
  store i32 0, ptr %5, align 4, !tbaa !1
  br label %12

12:                                               ; preds = %104, %11
  %13 = load i32, ptr %5, align 4, !tbaa !1
  %14 = icmp slt i32 %13, 10
  br i1 %14, label %16, label %15

15:                                               ; preds = %95, %86, %77, %68, %59, %50, %41, %32, %23, %12
  store i32 5, ptr %4, align 4
  call void @llvm.lifetime.end.p0(ptr %5) #4
  br label %106

16:                                               ; preds = %12
  %17 = load ptr, ptr %2, align 4, !tbaa !9
  %18 = load i32, ptr %3, align 4, !tbaa !1
  %19 = getelementptr inbounds [10 x i32], ptr %17, i32 %18
  %20 = getelementptr inbounds [10 x i32], ptr %19, i32 0, i32 %13
  %21 = load i32, ptr %20, align 4, !tbaa !1
  %22 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef %21)
  br label %23

23:                                               ; preds = %16
  %24 = add nsw i32 %13, 1
  store i32 %24, ptr %5, align 4, !tbaa !1
  %25 = load i32, ptr %5, align 4, !tbaa !1
  %26 = icmp slt i32 %25, 10
  br i1 %26, label %27, label %15

27:                                               ; preds = %23
  %28 = getelementptr inbounds [10 x i32], ptr %17, i32 %18
  %29 = getelementptr inbounds [10 x i32], ptr %28, i32 0, i32 %25
  %30 = load i32, ptr %29, align 4, !tbaa !1
  %31 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef %30)
  br label %32

32:                                               ; preds = %27
  %33 = add nsw i32 %25, 1
  store i32 %33, ptr %5, align 4, !tbaa !1
  %34 = load i32, ptr %5, align 4, !tbaa !1
  %35 = icmp slt i32 %34, 10
  br i1 %35, label %36, label %15

36:                                               ; preds = %32
  %37 = getelementptr inbounds [10 x i32], ptr %17, i32 %18
  %38 = getelementptr inbounds [10 x i32], ptr %37, i32 0, i32 %34
  %39 = load i32, ptr %38, align 4, !tbaa !1
  %40 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef %39)
  br label %41

41:                                               ; preds = %36
  %42 = add nsw i32 %34, 1
  store i32 %42, ptr %5, align 4, !tbaa !1
  %43 = load i32, ptr %5, align 4, !tbaa !1
  %44 = icmp slt i32 %43, 10
  br i1 %44, label %45, label %15

45:                                               ; preds = %41
  %46 = getelementptr inbounds [10 x i32], ptr %17, i32 %18
  %47 = getelementptr inbounds [10 x i32], ptr %46, i32 0, i32 %43
  %48 = load i32, ptr %47, align 4, !tbaa !1
  %49 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef %48)
  br label %50

50:                                               ; preds = %45
  %51 = add nsw i32 %43, 1
  store i32 %51, ptr %5, align 4, !tbaa !1
  %52 = load i32, ptr %5, align 4, !tbaa !1
  %53 = icmp slt i32 %52, 10
  br i1 %53, label %54, label %15

54:                                               ; preds = %50
  %55 = getelementptr inbounds [10 x i32], ptr %17, i32 %18
  %56 = getelementptr inbounds [10 x i32], ptr %55, i32 0, i32 %52
  %57 = load i32, ptr %56, align 4, !tbaa !1
  %58 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef %57)
  br label %59

59:                                               ; preds = %54
  %60 = add nsw i32 %52, 1
  store i32 %60, ptr %5, align 4, !tbaa !1
  %61 = load i32, ptr %5, align 4, !tbaa !1
  %62 = icmp slt i32 %61, 10
  br i1 %62, label %63, label %15

63:                                               ; preds = %59
  %64 = getelementptr inbounds [10 x i32], ptr %17, i32 %18
  %65 = getelementptr inbounds [10 x i32], ptr %64, i32 0, i32 %61
  %66 = load i32, ptr %65, align 4, !tbaa !1
  %67 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef %66)
  br label %68

68:                                               ; preds = %63
  %69 = add nsw i32 %61, 1
  store i32 %69, ptr %5, align 4, !tbaa !1
  %70 = load i32, ptr %5, align 4, !tbaa !1
  %71 = icmp slt i32 %70, 10
  br i1 %71, label %72, label %15

72:                                               ; preds = %68
  %73 = getelementptr inbounds [10 x i32], ptr %17, i32 %18
  %74 = getelementptr inbounds [10 x i32], ptr %73, i32 0, i32 %70
  %75 = load i32, ptr %74, align 4, !tbaa !1
  %76 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef %75)
  br label %77

77:                                               ; preds = %72
  %78 = add nsw i32 %70, 1
  store i32 %78, ptr %5, align 4, !tbaa !1
  %79 = load i32, ptr %5, align 4, !tbaa !1
  %80 = icmp slt i32 %79, 10
  br i1 %80, label %81, label %15

81:                                               ; preds = %77
  %82 = getelementptr inbounds [10 x i32], ptr %17, i32 %18
  %83 = getelementptr inbounds [10 x i32], ptr %82, i32 0, i32 %79
  %84 = load i32, ptr %83, align 4, !tbaa !1
  %85 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef %84)
  br label %86

86:                                               ; preds = %81
  %87 = add nsw i32 %79, 1
  store i32 %87, ptr %5, align 4, !tbaa !1
  %88 = load i32, ptr %5, align 4, !tbaa !1
  %89 = icmp slt i32 %88, 10
  br i1 %89, label %90, label %15

90:                                               ; preds = %86
  %91 = getelementptr inbounds [10 x i32], ptr %17, i32 %18
  %92 = getelementptr inbounds [10 x i32], ptr %91, i32 0, i32 %88
  %93 = load i32, ptr %92, align 4, !tbaa !1
  %94 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef %93)
  br label %95

95:                                               ; preds = %90
  %96 = add nsw i32 %88, 1
  store i32 %96, ptr %5, align 4, !tbaa !1
  %97 = load i32, ptr %5, align 4, !tbaa !1
  %98 = icmp slt i32 %97, 10
  br i1 %98, label %99, label %15

99:                                               ; preds = %95
  %100 = getelementptr inbounds [10 x i32], ptr %17, i32 %18
  %101 = getelementptr inbounds [10 x i32], ptr %100, i32 0, i32 %97
  %102 = load i32, ptr %101, align 4, !tbaa !1
  %103 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef %102)
  br label %104

104:                                              ; preds = %99
  %105 = add nsw i32 %97, 1
  store i32 %105, ptr %5, align 4, !tbaa !1
  br label %12, !llvm.loop !27

106:                                              ; preds = %15
  %107 = call i32 (ptr, ...) @printf(ptr noundef @.str.2)
  br label %108

108:                                              ; preds = %106
  %109 = load i32, ptr %3, align 4, !tbaa !1
  %110 = add nsw i32 %109, 1
  store i32 %110, ptr %3, align 4, !tbaa !1
  br label %7, !llvm.loop !28

111:                                              ; preds = %10
  %112 = call i32 (ptr, ...) @printf(ptr noundef @.str)
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
!9 = !{!10, !10, i64 0}
!10 = !{!"p1 int", !11, i64 0}
!11 = !{!"any pointer", !3, i64 0}
!12 = distinct !{!12, !13, !14}
!13 = !{!"llvm.loop.mustprogress"}
!14 = !{!"llvm.loop.unroll.disable"}
!15 = !{!16, !2, i64 4}
!16 = !{!"config", !2, i64 0, !2, i64 4}
!17 = distinct !{!17, !13}
!18 = distinct !{!18, !13, !14}
!19 = !{!16, !2, i64 0}
!20 = distinct !{!20, !13}
!21 = distinct !{!21, !13}
!22 = distinct !{!22, !13}
!23 = distinct !{!23, !13}
!24 = distinct !{!24, !13}
!25 = distinct !{!25, !13}
!26 = distinct !{!26, !13}
!27 = distinct !{!27, !13, !14}
!28 = distinct !{!28, !13}
