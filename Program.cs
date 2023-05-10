using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using System;
using System.Diagnostics;
using System.Runtime.ConstrainedExecution;

namespace Apex.CUDA
{
    class Program
    {
        public static void Main(string[] args)
        {
            //BottleNeckTest();


            Console.WriteLine("{0,-20} {1,-20} {2,-20} {3,-20} {4,-20}\n", "Length", "SingleThread", "TPL", "CLA", "CUDA");
            //TestStart(10);
            //TestStart(1_000);
            //TestStart(10_000);
            //TestStart(100_000);
            //TestStart(1_000_000);
            //TestStart(10_000_000);
            //TestStart(100_000_000);
            //TestStart(1_000_000_000);
            TestStart(2_146_435_071);

            Console.WriteLine("Done, Press any key to exit");
            Console.ReadKey();
        }

        public static void TestStart(long length)
        {
            var lengthString = length.ToString("#,##0");

            List<AcceleratorType> accTypes = ListGPU();

            if (accTypes.Contains(AcceleratorType.OpenCL) && accTypes.Contains(AcceleratorType.Cuda))
            {
                Console.WriteLine("{0,-20} {1,-20} {2,-20} {3,-20} {4,-20}", lengthString, SingleThread(length), Tpl(length), CLA(length), CUDA(length));
            }
            else if (accTypes.Contains(AcceleratorType.OpenCL))
            {
                Console.WriteLine("{0,-20} {1,-20} {2,-20} {3,-20} {4,-20}", lengthString, SingleThread(length), Tpl(length), CLA(length), "N/A");
            }
            else if (accTypes.Contains(AcceleratorType.Cuda))
            {
                Console.WriteLine("{0,-20} {1,-20} {2,-20} {3,-20} {4,-20}", lengthString, SingleThread(length), Tpl(length), "N/A", CLA(length));
            }
            else
            {
                Console.WriteLine("{0,-20} {1,-20} {2,-20} {3,-20} {4,-20}", lengthString, SingleThread(length), Tpl(length), "N/A", "N/A");
            }
        }

        public static List<AcceleratorType> ListGPU()
        {
            Context con = Context.Create(builder => builder.AllAccelerators());

            // Get a list of all available accelerators (including the CPU)
            // CPUAccelerator[Type: CPU, WarpSize: 4, MaxNumThreadsPerGroup: 16, MemorySize: 9223372036854775807]
            // Intel(R) HD Graphics 630[Type: OpenCL, WarpSize: 0, MaxNumThreadsPerGroup: 256, MemorySize: 13557809152]
            // NVIDIA GeForce GTX 1050[Type: Cuda, WarpSize: 32, MaxNumThreadsPerGroup: 1024, MemorySize: 4294836224]

            List<AcceleratorType> acceleratorTypes = new List<AcceleratorType>();
            foreach (Device device in con)
            {
                acceleratorTypes.Add(device.AcceleratorType);
            }

            return acceleratorTypes;
        }

        public static string CLA(long length)
        {
            String result = "";

            Context context = Context.CreateDefault();

            Accelerator acceleratorCLA = context.CreateCLAccelerator(0);
            result = result + Gpu(acceleratorCLA, length);
            acceleratorCLA.Dispose();


            context.Dispose();

            return result;

        }
        public static string CUDA(long length)
        {
            String result = "";

            Context context = Context.CreateDefault();

            Accelerator acceleratorCUDA = context.CreateCudaAccelerator(0);
            result = result + Gpu(acceleratorCUDA, length);
            acceleratorCUDA.Dispose();

            context.Dispose();

            return result;
        }

        public static string Gpu(Accelerator accelerator, long length)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            // Allocate memory on the accelerator.
            var deviceOutput = accelerator.Allocate1D<long>(length);

            // Load / Compile the kernel. This is where the magic happens.
            var loadedKernel = accelerator.LoadAutoGroupedStreamKernel(
            (Index1D i, ArrayView<long> output) =>
                {
                    output[i] = 5 * 5 * 5 * 5 * 5;
                });

            // Tell the accelerator to start computing the kernel
            loadedKernel((int)deviceOutput.Length, deviceOutput.View);

            // Wait for the accelerator to be finished with whatever it's doing
            // in this case it just waits for the kernel to finish.
            accelerator.Synchronize();

            var result = deviceOutput.GetAsArray1D();

            return stopwatch.ElapsedMilliseconds.ToString("#,##0");
        }

        public static string Tpl(long length)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();


            long[] output = new long[length];
            Parallel.For(0, output.Length,
            (long i) =>
            {
                output[i] = 5 * 5 * 5 * 5 * 5;
            });

            return stopwatch.ElapsedMilliseconds.ToString("#,##0");
        }

        public static string SingleThread(long length)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            long[] output = new long[length];
            for (long i = 0; i < output.Length; i++)
            {
                output[i] = 5 * 5 * 5 * 5 * 5;
            }

            return stopwatch.ElapsedMilliseconds.ToString("#,##0");
        }



        // Test to see which calls takes the most amount of time
        public static void BottleNeckTest()
        {

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            long length = 1_000_000_100;

            // Initialize ILGPU
            Context context = Context.CreateDefault();
            //Accelerator accelerator = context.CreateCudaAccelerator(0);
            Accelerator accelerator = context.CreateCLAccelerator(0);

            Console.WriteLine("Initialize ILGPU Took : " + stopwatch.ElapsedTicks.ToString("#,##0"));
            stopwatch.Restart();

            // Allocate memory on the accelerator.
            var deviceOutput = accelerator.Allocate1D<int>(length);

            // Load / Compile the kernel. This is where the magic happens.
            var loadedKernel = accelerator.LoadAutoGroupedStreamKernel(
            (Index1D i, ArrayView<int> output) =>
            {
                output[i] = 5 * 5 * 5 * 5 * 5;
            });

            Console.WriteLine("Load Kernal Took : " + stopwatch.ElapsedTicks.ToString("#,##0"));
            stopwatch.Restart();

            // Tell the accelerator to start computing the kernel
            loadedKernel((int)deviceOutput.Length, deviceOutput.View);
            Console.WriteLine("Computing Took :" + stopwatch.ElapsedTicks.ToString("#,##0"));
            stopwatch.Restart();

            // Wait for the accelerator to be finished with whatever it's doing in this case it just waits for the kernel to finish.
            accelerator.Synchronize();
            var result = deviceOutput.GetAsArray1D();
            context.Dispose();
            Console.WriteLine("Synchronize and Dispose Took :" + stopwatch.ElapsedTicks.ToString("#,##0"));
        }

    }
}