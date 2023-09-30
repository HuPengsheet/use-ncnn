​	ncnn里面主要有三种内存申请的方式，分别是ncnn::fastMalloc、PoolAllocator和UnlockedPoolAllocator。其中PoolAllocator和UnlockedPoolAllocator里会调用ncnn::fastMalloc分配内存，然后作为内存池来维护内存。PoolAllocator和UnlockedPoolAllocator分别是带锁的内存池和不带锁的内存池。下面展开来讲：

## ncnn::fastMalloc与ncnn::fastFree

```c++
//src/allocator.h

static NCNN_FORCEINLINE void* fastMalloc(size_t size)
{
#if _MSC_VER
    return _aligned_malloc(size, NCNN_MALLOC_ALIGN);
#elif (defined(__unix__) || defined(__APPLE__)) && _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
    void* ptr = 0;
    if (posix_memalign(&ptr, NCNN_MALLOC_ALIGN, size + NCNN_MALLOC_OVERREAD))
        ptr = 0;
    return ptr;
#elif __ANDROID__ && __ANDROID_API__ < 17
    return memalign(NCNN_MALLOC_ALIGN, size + NCNN_MALLOC_OVERREAD);
#else
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + NCNN_MALLOC_ALIGN + NCNN_MALLOC_OVERREAD);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, NCNN_MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
#endif
}

static NCNN_FORCEINLINE void fastFree(void* ptr)
{
    if (ptr)
    {
#if _MSC_VER
        _aligned_free(ptr);
#elif (defined(__unix__) || defined(__APPLE__)) && _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
        free(ptr);
#elif __ANDROID__ && __ANDROID_API__ < 17
        free(ptr);
#else
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
#endif
    }
}
```

看起来很复杂是吧，其实把一些宏去掉看起来就很简单了，在linux平台下实际就是调用如下函数：

```c++
static NCNN_FORCEINLINE void* fastMalloc(size_t size)
{
    void* ptr = 0;
    if (posix_memalign(&ptr, NCNN_MALLOC_ALIGN, size + NCNN_MALLOC_OVERREAD))
        ptr = 0;
    return ptr;
}
static NCNN_FORCEINLINE void fastFree(void* ptr)
{
    if (ptr)
    {
        free(ptr);
    }
}
```

​	也就是调用stdlib里面的posix_memalign函数来分配内存，下面是posix_memalign的函数原型：

```c++
int posix_memalign(void **memptr，size_t alignment，size_t size);
```

​	调用posix_memalign( )成功时会返回size字节的动态内存，并且这块内存的地址是alignment的倍数。参数alignment必须是2的幂，还是void指针的大小的倍数。

```c++
// the alignment of all the allocated buffers
#if NCNN_AVX512
#define NCNN_MALLOC_ALIGN 64
#elif NCNN_AVX
#define NCNN_MALLOC_ALIGN 32
#else
#define NCNN_MALLOC_ALIGN 16
#endif

// we have some optimized kernels that may overread buffer a bit in loop
// it is common to interleave next-loop data load with arithmetic instructions
// allocating more bytes keeps us safe from SEGV_ACCERR failure
#define NCNN_MALLOC_OVERREAD 64
```

​	使用fastMalloc分配内存，会默认多分配NCNN_MALLOC_OVERREAD个字节，也就是64个字节，这是因为有一些优化的内核，它们可能会在循环中稍微溢出缓冲区，避免SEGV_ACCERR故障的影响。

​	fastFree就没什么好将的啦，就是释放内存。

## PoolAllocator和UnlockedPoolAllocator

​	PoolAllocator和UnlockedPoolAllocator都是使用内存池管理内存，都继承于一个虚类：

```c++
class NCNN_EXPORT Allocator
{
public:
    virtual ~Allocator();
    virtual void* fastMalloc(size_t size) = 0;
    virtual void fastFree(void* ptr) = 0;
};
```

​	PoolAllocator和UnlockedPoolAllocator的区别是带不带锁，ncnn默认使用的是带锁的PoolAllocator，但两者的结构和原理都是一样的，这里为了讲解方便，我们对UnlockedPoolAllocator展开描述。

```c++
class UnlockedPoolAllocatorPrivate
{
public:
    unsigned int size_compare_ratio; // 0~256
    size_t size_drop_threshold;
    std::list<std::pair<size_t, void*> > budgets;
    std::list<std::pair<size_t, void*> > payouts;
};

class NCNN_EXPORT UnlockedPoolAllocator : public Allocator
{
public:
    UnlockedPoolAllocator();
    ~UnlockedPoolAllocator();

    // ratio range 0 ~ 1
    // default cr = 0
    void set_size_compare_ratio(float scr);

    // budget drop threshold
    // default threshold = 10
    void set_size_drop_threshold(size_t);

    // release all budgets immediately
    void clear();

    virtual void* fastMalloc(size_t size);
    virtual void fastFree(void* ptr);

private:
    UnlockedPoolAllocator(const UnlockedPoolAllocator&);
    UnlockedPoolAllocator& operator=(const UnlockedPoolAllocator&);

private:
    UnlockedPoolAllocatorPrivate* const d;
};
```

​	类的结构也比较简单，实际上就是在UnlockedPoolAllocator中的UnlockedPoolAllocatorPrivate里维护了两个list，每个list里的元素`std::pair<size_t, void*>`记录内存的大小和地址。budgets里记录的是空闲的内存，payouts里记录的是正在使用的内存。

​	我们接下来看这个内存池的管理，也就是fastMalloc和fastFree的实现：

### UnlockedPoolAllocator的fastMalloc

```c++
void* UnlockedPoolAllocator::fastMalloc(size_t size)
{
    // find free budget
    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin(), it_max = d->budgets.begin(), it_min = d->budgets.begin();
    for (; it != d->budgets.end(); ++it)
    {
        size_t bs = it->first;

        // size_compare_ratio ~ 100%
        if (bs >= size && ((bs * d->size_compare_ratio) >> 8) <= size)
        {
            void* ptr = it->second;

            d->budgets.erase(it);

            d->payouts.push_back(std::make_pair(bs, ptr));

            return ptr;
        }

        if (bs > it_max->first)
        {
            it_max = it;
        }
        if (bs < it_min->first)
        {
            it_min = it;
        }
    }

    if (d->budgets.size() >= d->size_drop_threshold)
    {
        if (it_max->first < size)
        {
            ncnn::fastFree(it_min->second);
            d->budgets.erase(it_min);
        }
        else if (it_min->first > size)
        {
            ncnn::fastFree(it_max->second);
            d->budgets.erase(it_max);
        }
    }

    // new
    void* ptr = ncnn::fastMalloc(size);

    d->payouts.push_back(std::make_pair(size, ptr));

    return ptr;
}
```

​	给大家分析一下这个内存池的代码：
```c++
for (; it != d->budgets.end(); ++it)
{
    size_t bs = it->first;

    // size_compare_ratio ~ 100%
    if (bs >= size && ((bs * d->size_compare_ratio) >> 8) <= size)
    {
        void* ptr = it->second;

        d->budgets.erase(it);

        d->payouts.push_back(std::make_pair(bs, ptr));

        return ptr;
    }

    if (bs > it_max->first)
    {
        it_max = it;
    }
    if (bs < it_min->first)
    {
        it_min = it;
    }
}
```

​	这段代码是遍历内存池中所有没有被占用的内存块，当有空闲内存块满足`bs >= size && ((bs * d->size_compare_ratio) >> 8) <= size`，也就是这个内存块的大小即比你要的大，但是也不会大的特别多（**假如你要1个字节，我分配100字节的内存块话那不就是浪费了**），那就直接返回这个内存块的指针。

​	如果没有找到合适大小的内存块的话`it_max`和`it_min`就会存储当前空闲内存块中内存最大和最小的迭代器位置。

```c++
if (d->budgets.size() >= d->size_drop_threshold)
{
    if (it_max->first < size)
    {
        ncnn::fastFree(it_min->second);
        d->budgets.erase(it_min);
    }
    else if (it_min->first > size)
    {
        ncnn::fastFree(it_max->second);
        d->budgets.erase(it_max);
    }
}
```

​	如果没有合适的内存的话，说明要么是空闲的内存块太大或者空闲的内存块太小，当`d->budgets.size() >= d->size_drop_threshold`其中ncnn设置d->size_drop_threshold为10，也就是内存池中已经大于等于10个空闲内存块了，但是没有符合我们大小要求的，所以此时会对内存块进行擦除，如果最大得内存块都不够大，就去除最小得内存块，相反，如果最小得内存块都太大了，就去除最大得内存块。

```c++
void* ptr = ncnn::fastMalloc(size);

d->payouts.push_back(std::make_pair(size, ptr));

return ptr;
```

​	使用我们上面讲的ncnn::fastMalloc分配内存，在payouts插入信息，表明这块内存占用

### UnlockedPoolAllocator的fastFree

```c++
void UnlockedPoolAllocator::fastFree(void* ptr)
{
    // return to budgets
    std::list<std::pair<size_t, void*> >::iterator it = d->payouts.begin();
    for (; it != d->payouts.end(); ++it)
    {
        if (it->second == ptr)
        {
            size_t size = it->first;

            d->payouts.erase(it);

            d->budgets.push_back(std::make_pair(size, ptr));

            return;
        }
    }

    NCNN_LOGE("FATAL ERROR! unlocked pool allocator get wild %p", ptr);
    ncnn::fastFree(ptr);
}
```

​	释放的代码很简单，遍历payouts，找到对应的那个节点，然后把它删除，并把它放入budgets中变成了空闲内存块。