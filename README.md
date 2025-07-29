# Current Limitations
- Splitting: How do we handle splits based on different metadata targets?

# Benchmarking
| Dataset       | Target        | Task           | Level   | N   |
|---------------|---------------|----------------|---------|-----|
| Danenberg2022 | cell_type     | regression     | patch   | 693 |
| Danenberg2022 | follow-up     | regression     | patient | 693 |
| Danenberg2022 | IntClust      | classification | patient | 693 |
| Danenberg2022 | tumor/normal  | classification | patient | 693 |
| PCa2025       | cell_type     | regression     | patch   | 153 |
| PCa2025       | follow-up     | regression     | patient | 153 |
| PCa2025       | disease_progr | classification | patient | 153 |
| PCa2025       | grade         | classification | patient | 153 |
| BLCa2025      | cell_type     | regression     | patch   |  |
