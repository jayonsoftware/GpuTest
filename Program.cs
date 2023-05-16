using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using System.Diagnostics;


namespace Apex.CUDA
{
    class Program
    {
        public static void Main(string[] args)
        {
            //BottleNeckTest();

            PrintGpuDetails();

            Console.WriteLine("{0,-20} {1,-20} {2,-20} {3,-20} {4,-20}\n", "Length", "SingleThread", "TPL", "CLA", "CUDA");
            //TestStart(10);
            //TestStart(1_000);
            //TestStart(10_000);
            //TestStart(100_000);
            //TestStart(1_000_000);
            //TestStart(10_000_000);
            //TestStart(100_000_000);
            TestStart(1_000_000_000);
            //TestStart(2_146_435_071);

            Console.WriteLine("Done, Press any key to exit");
            //Console.ReadKey();
        }

        public static void TestStart(long length)
        {
            var lengthString = length.ToString("#,##0");

            Context con = Context.Create(builder => builder.AllAccelerators());

            // Get a list of all available accelerators (including the CPU/GPU)
            List<AcceleratorType> accTypes = new List<AcceleratorType>();
            foreach (Device device in con)
            {
                accTypes.Add(device.AcceleratorType);
            }

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
                Console.WriteLine("{0,-20} {1,-20} {2,-20} {3,-20} {4,-20}", lengthString, SingleThread(length), Tpl(length), "N/A", CUDA(length));
            }
            else
            {
                Console.WriteLine("{0,-20} {1,-20} {2,-20} {3,-20} {4,-20}", lengthString, SingleThread(length), Tpl(length), "N/A", "N/A");
            }
        }

        public static void PrintGpuDetails()
        {
            Context acceleratorContext = Context.Create(builder => builder.AllAccelerators());

            foreach (var accelerator in acceleratorContext)
            {
                Console.WriteLine($"Name: {accelerator.Name}");
                Console.WriteLine($"MemorySize: {accelerator.MemorySize}");
                Console.WriteLine($"MaxThreadsPerGroup: {accelerator.MaxNumThreadsPerGroup}");
                Console.WriteLine($"MaxSharedMemoryPerGroup: {accelerator.MaxSharedMemoryPerGroup}");
                Console.WriteLine($"MaxGridSize: {accelerator.MaxGridSize}");
                Console.WriteLine($"MaxConstantMemory: {accelerator.MaxConstantMemory}");
                Console.WriteLine($"WarpSize: {accelerator.WarpSize}");
                Console.WriteLine($"NumMultiprocessors: {accelerator.NumMultiprocessors}");
                Console.WriteLine();
            }

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
            var deviceOutput = accelerator.Allocate1D<double>(length);

            // Load / Compile the kernel. This is where the magic happens.
            var loadedKernel = accelerator.LoadAutoGroupedStreamKernel(
            (Index1D i, ArrayView<double> output) =>
                {
                    output[i] = 5.2429049278429 * 5.2429049278429 * 5.2429049278429 * 5.2429049278429 * 5.2429049278429;
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


            double[] output = new double[length];
            Parallel.For(0, output.Length,
            (long i) =>
            {
                output[i] = 5.2429049278429 * 5.2429049278429 * 5.2429049278429 * 5.2429049278429 * 5.2429049278429;
            });

            return stopwatch.ElapsedMilliseconds.ToString("#,##0");
        }

        public static string SingleThread(long length)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            double[] output = new double[length];
            for (int i = 0; i < output.Length; i++)
            {
                output[i] = 5.2429049278429 * 5.2429049278429 * 5.2429049278429 * 5.2429049278429 * 5.2429049278429;
            }

            return stopwatch.ElapsedMilliseconds.ToString("#,##0");
        }
    }
}