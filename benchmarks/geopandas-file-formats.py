import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import time
import os
import tempfile
import shutil
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class GeoBenchmark:
    def __init__(self, n_points=1e7, height: int = 1000, width: int = 1000, output_dir=None):
        """
        Initialize the benchmark with specified number of points

        Parameters:
        -----------
        n_points : int
            Number of points to generate for testing
        output_dir : str or Path
            Directory to save test files (uses temp dir if None)
        """
        self.n_points = n_points
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        self.output_dir.mkdir(exist_ok=True)
        self.height, self.width = height, width
        self.results = []

    def generate_test_data(self):
        """Generate random point data for testing"""
        print(f"Generating {self.n_points:,} random points...")

        # Generate random coordinates (roughly covering a country-sized area)
        np.random.seed(42)  # For reproducible results
        lons = np.random.uniform(0, self.height, self.n_points)
        lats = np.random.uniform(0, self.width, self.n_points)

        # Create some additional attributes
        ids = np.arange(self.n_points)
        categories = np.random.choice(['A', 'B', 'C', 'D'], self.n_points)
        values = np.random.normal(100, 25, self.n_points)

        # Create GeoDataFrame
        geometry = gpd.points_from_xy(lons, lats)
        self.gdf = gpd.GeoDataFrame({
            'id': ids,
            'category': categories,
            'value': values,
            'geometry': geometry
        }, crs='EPSG:4326')

        print("Test data generated successfully!")
        return self.gdf

    def save_formats(self):
        """Save the data in different formats"""
        formats = {
            'parquet': {'file': 'test_data.parquet', 'method': 'to_parquet'},
            'feather': {'file': 'test_data.feather', 'method': 'to_feather'},
            'gpkg': {'file': 'test_data.gpkg', 'method': 'to_file'},
            'shp': {'file': 'test_data.shp', 'method': 'to_file'},
            'geojson': {'file': 'test_data.geojson', 'method': 'to_file'},
        }

        print("Saving data in different formats...")

        for fmt, info in formats.items():
            filepath = self.output_dir / info['file']
            start_time = time.time()

            try:
                if fmt == 'parquet':
                    self.gdf.to_parquet(filepath)
                elif fmt == 'feather':
                    self.gdf.to_feather(filepath)
                elif fmt in ['gpkg', 'shp', 'geojson']:
                    driver = {'gpkg': 'GPKG', 'shp': 'ESRI Shapefile', 'geojson': 'GeoJSON'}
                    self.gdf.to_file(filepath, driver=driver[fmt])

                save_time = time.time() - start_time
                file_size = filepath.stat().st_size / (1024 * 1024)  # Size in MB

                print(f"  {fmt.upper()}: {save_time:.2f}s, {file_size:.1f}MB")

            except Exception as e:
                print(f"  {fmt.upper()}: Failed - {e}")

    def benchmark_bbox_query(self, bbox, description=""):
        """
        Benchmark bounding box queries across different formats

        Parameters:
        -----------
        bbox : tuple
            Bounding box as (minx, miny, maxx, maxy)
        description : str
            Description of the bounding box for reporting
        """
        minx, miny, maxx, maxy = bbox

        formats_to_test = [
            ('parquet', 'test_data.parquet'),
            ('feather', 'test_data.feather'),
            ('gpkg', 'test_data.gpkg'),
            ('shp', 'test_data.shp'),
            ('geojson', 'test_data.geojson'),
        ]

        print(f"\nBenchmarking bounding box query{' (' + description + ')' if description else ''}:")
        print(f"BBox: {bbox}")

        bbox_results = []

        for fmt, filename in formats_to_test:
            filepath = self.output_dir / filename

            if not filepath.exists():
                print(f"  {fmt.upper()}: File not found, skipping")
                continue

            times = []
            result_counts = []

            # Run multiple iterations for more stable timing
            for i in range(3):
                start_time = time.time()

                try:
                    if fmt in ['parquet', 'feather']:
                        # For parquet/feather, load all data then filter
                        gdf = gpd.read_parquet(filepath) if fmt == 'parquet' else gpd.read_feather(filepath)
                        # Filter by bounding box
                        mask = (
                                (gdf.geometry.x >= minx) & (gdf.geometry.x <= maxx) &
                                (gdf.geometry.y >= miny) & (gdf.geometry.y <= maxy)
                        )
                        result = gdf[mask]
                    else:
                        # For file formats that support spatial indexing
                        result = gpd.read_file(filepath, bbox=tuple(bbox))

                    query_time = time.time() - start_time
                    times.append(query_time)
                    result_counts.append(len(result))

                except Exception as e:
                    print(f"  {fmt.upper()}: Error - {e}")
                    break

            if times:
                avg_time = np.mean(times)
                std_time = np.std(times)
                avg_count = np.mean(result_counts)

                print(f"  {fmt.upper()}: {avg_time:.3f}s ± {std_time:.3f}s, {int(avg_count):,} points")

                bbox_results.append({
                    'format': fmt,
                    'bbox_description': description,
                    'bbox': bbox,
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'result_count': int(avg_count),
                    'points_per_second': int(avg_count / avg_time) if avg_time > 0 else 0
                })

        self.results.extend(bbox_results)
        return bbox_results

    def benchmark_spatial_index_creation(self):
        """Benchmark spatial index creation for supported formats"""
        print("\nTesting spatial index creation (where supported):")

        # Test with GeoPackage (supports spatial indexing)
        gpkg_file = self.output_dir / 'test_data.gpkg'
        if gpkg_file.exists():
            start_time = time.time()
            gdf = gpd.read_file(gpkg_file)
            # Create spatial index
            gdf.sindex
            index_time = time.time() - start_time
            print(f"  GPKG spatial index: {index_time:.3f}s")

    def run_comprehensive_benchmark(self):
        """Run a comprehensive benchmark with different bounding box sizes"""
        # Generate test data
        self.generate_test_data()

        # Save in different formats
        self.save_formats()

        # Test different bounding box sizes
        bboxes = [
            # Small bbox (roughly 1% of data)
            (-1, 49, 1, 51, "Small (1% of data)"),
            # Medium bbox (roughly 10% of data)
            (-3, 47, 3, 53, "Medium (10% of data)"),
            # Large bbox (roughly 50% of data)
            (-6, 45, 6, 55, "Large (50% of data)"),
        ]

        for *bbox, description in bboxes:
            self.benchmark_bbox_query(bbox, description)

        # Test spatial indexing
        self.benchmark_spatial_index_creation()

        return self.results

    def print_summary(self):
        """Print a summary of benchmark results"""
        if not self.results:
            print("No results to summarize")
            return

        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        # Group by bbox description
        bbox_groups = {}
        for result in self.results:
            desc = result['bbox_description']
            if desc not in bbox_groups:
                bbox_groups[desc] = []
            bbox_groups[desc].append(result)

        for desc, group in bbox_groups.items():
            print(f"\n{desc}:")
            print("-" * 40)

            # Sort by average time
            sorted_group = sorted(group, key=lambda x: x['avg_time'])

            for result in sorted_group:
                print(f"{result['format'].upper():>8}: "
                      f"{result['avg_time']:.3f}s, "
                      f"{result['result_count']:,} points, "
                      f"{result['points_per_second']:,} pts/s")

            # Show fastest format
            if sorted_group:
                fastest = sorted_group[0]
                print(f"         → Fastest: {fastest['format'].upper()}")

    def save_results_csv(self, filename="benchmark_results.csv"):
        """Save results to CSV file"""
        if self.results:
            df = pd.DataFrame(self.results)
            csv_path = self.output_dir / filename
            df.to_csv(csv_path, index=False)
            print(f"\nResults saved to: {csv_path}")

    def cleanup(self):
        """Clean up temporary files"""
        if self.output_dir.name.startswith('tmp'):
            shutil.rmtree(self.output_dir)
            print(f"Cleaned up temporary directory: {self.output_dir}")


def main():
    """Main function to run the benchmark"""
    print("Geopandas Bounding Box Query Benchmark")
    print("=" * 50)

    # You can adjust these parameters
    N_POINTS = int(1e7)  # Start with smaller number for testing
    OUTPUT_DIR = "benchmark_results"  # None for temp directory

    benchmark = GeoBenchmark(n_points=N_POINTS, output_dir=OUTPUT_DIR)

    try:
        # Run the benchmark
        results = benchmark.run_comprehensive_benchmark()

        # Print summary
        benchmark.print_summary()

        # Save results
        benchmark.save_results_csv()

        print(f"\nBenchmark completed! Files saved in: {benchmark.output_dir}")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Uncomment the next line if you want to auto-cleanup temp files
        # benchmark.cleanup()
        pass


if __name__ == "__main__":
    main()
