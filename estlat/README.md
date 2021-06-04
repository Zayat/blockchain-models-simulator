# About

Estimate “best-case” latencies between any two geographic locations.

The script `estlat.py` expects two arguments---a file containing locations and another for persisting the output. The “locations file” is a space-delimited file with each line providing geographic location of one specific location and contains three columns, namely “location name”, “latitude”, and “longitude.”

## Usage

```
§ ./bin/estlat.py
Usage: ./bin/estlat.py <cities-file> <out-file>
```

## Example

```
§ ./bin/estlat.py test/cities-2.txt test/adjlst-2.txt
```

You may also add a new locations file (named following the format of “cities-*.txt”) in the `test/` directory and run `make` from same directory to generate the corresponding adjacency-list file (following the format “adjlst-*.txt”).
